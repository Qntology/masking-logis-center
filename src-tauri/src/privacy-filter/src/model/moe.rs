/// Sparse Mixture-of-Experts matching the Python OpenAIPrivacyFilterMLP exactly.
///
/// Key details:
///   - Router: Linear [hidden_size, num_experts] with bias, top-k selection
///   - Routing weights: softmax(top_k_logits) / top_k
///   - Expert gating: custom with alpha=1.702, limit=7.0
///     gate, up = gate_up.chunk(2)
///     gate = clamp(gate, max=7.0)
///     up = clamp(up, -7.0, 7.0)
///     glu = gate * sigmoid(gate * 1.702)
///     out = (up + 1) * glu
///   - All computation in fp32
///   - Output multiplied by num_experts_per_tok

use burn::prelude::*;
use burn::module::{Param, ParamId};

const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;

#[derive(Debug)]
pub struct SparseMoE<B: Backend> {
    // Router (used as Burn tensors for BLAS matmul)
    pub router_weight: Param<Tensor<B, 2>>,  // [hidden_size, num_experts]
    pub router_bias: Param<Tensor<B, 1>>,    // [num_experts]

    // Expert weights — stored as Burn tensors for loading, but also cached as
    // flat Vec<f32> for fast per-expert slicing during forward.
    pub gate_up_proj: Param<Tensor<B, 3>>,       // [num_experts, hidden_size, 2*intermediate_size]
    pub gate_up_proj_bias: Param<Tensor<B, 2>>,  // [num_experts, 2*intermediate_size]
    pub down_proj: Param<Tensor<B, 3>>,          // [num_experts, intermediate_size, hidden_size]
    pub down_proj_bias: Param<Tensor<B, 2>>,     // [num_experts, hidden_size]

    // CPU caches — populated by `cache_weights()` after loading.
    pub gu_cache: Vec<f32>,
    pub gu_b_cache: Vec<f32>,
    pub dp_cache: Vec<f32>,
    pub dp_b_cache: Vec<f32>,

    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl<B: Backend> SparseMoE<B> {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        device: &B::Device,
    ) -> Self {
        let router_weight = Tensor::zeros([hidden_size, num_experts], device);
        let router_bias = Tensor::zeros([num_experts], device);

        let gate_up_proj = Tensor::zeros([num_experts, hidden_size, 2 * intermediate_size], device);
        let gate_up_proj_bias = Tensor::zeros([num_experts, 2 * intermediate_size], device);
        let down_proj = Tensor::zeros([num_experts, intermediate_size, hidden_size], device);
        let down_proj_bias = Tensor::zeros([num_experts, hidden_size], device);

        Self {
            router_weight: Param::initialized(ParamId::new(), router_weight),
            router_bias: Param::initialized(ParamId::new(), router_bias),
            gate_up_proj: Param::initialized(ParamId::new(), gate_up_proj),
            gate_up_proj_bias: Param::initialized(ParamId::new(), gate_up_proj_bias),
            down_proj: Param::initialized(ParamId::new(), down_proj),
            down_proj_bias: Param::initialized(ParamId::new(), down_proj_bias),
            gu_cache: Vec::new(),
            gu_b_cache: Vec::new(),
            dp_cache: Vec::new(),
            dp_b_cache: Vec::new(),
            num_experts,
            num_experts_per_tok,
            hidden_size,
            intermediate_size,
        }
    }

    /// Cache expert weight data as flat CPU vecs for fast per-expert slicing.
    /// Call once after weights are loaded.
    pub fn cache_weights(&mut self) {
        self.gu_cache = self.gate_up_proj.val().clone().to_data().convert::<f32>().to_vec::<f32>().unwrap();
        self.gu_b_cache = self.gate_up_proj_bias.val().clone().to_data().convert::<f32>().to_vec::<f32>().unwrap();
        self.dp_cache = self.down_proj.val().clone().to_data().convert::<f32>().to_vec::<f32>().unwrap();
        self.dp_b_cache = self.down_proj_bias.val().clone().to_data().convert::<f32>().to_vec::<f32>().unwrap();
    }

    /// Forward pass for MoE.
    ///
    /// Input: [batch, seq_len, hidden_size]
    /// Output: [batch, seq_len, hidden_size]
    pub fn forward(&self, hidden_states: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 3> {
        let [batch, seq_len, hidden_size] = hidden_states.dims();
        let total_tokens = batch * seq_len;
        let top_k = self.num_experts_per_tok;
        let inter2 = 2 * self.intermediate_size;

        // Flatten to [total_tokens, hidden_size]
        let flat = hidden_states.reshape([total_tokens, hidden_size]);

        // Router: BLAS matmul
        let router_logits = flat.clone().matmul(self.router_weight.val().clone())
            + self.router_bias.val().clone().unsqueeze_dim(0);

        // Top-k selection on CPU
        let router_data: Vec<f32> = router_logits.to_data().convert::<f32>().to_vec::<f32>().unwrap();

        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![vec![]; self.num_experts];

        for t in 0..total_tokens {
            let logits = &router_data[t * self.num_experts..(t + 1) * self.num_experts];

            let mut indexed: Vec<(usize, f32)> = logits.iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let max_val = indexed[0].1;
            let exp_sum: f32 = indexed[..top_k].iter().map(|&(_, v)| (v - max_val).exp()).sum();

            for k in 0..top_k {
                let (expert_idx, val) = indexed[k];
                let weight = (val - max_val).exp() / exp_sum / top_k as f32;
                expert_assignments[expert_idx].push((t, weight));
            }
        }

        // Expert computation using cached CPU weight data + BLAS matmul
        let flat_data: Vec<f32> = flat.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        let mut result_data = vec![0f32; total_tokens * hidden_size];

        let gu_stride = hidden_size * inter2;
        let dp_stride = self.intermediate_size * hidden_size;

        for (eidx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let n = assignments.len();

            // Gather input tokens
            let mut input_data = vec![0f32; n * hidden_size];
            for (i, &(tidx, _)) in assignments.iter().enumerate() {
                input_data[i * hidden_size..(i + 1) * hidden_size]
                    .copy_from_slice(&flat_data[tidx * hidden_size..(tidx + 1) * hidden_size]);
            }

            // Slice expert weights from cached CPU data (fast memcpy)
            let gu_w_start = eidx * gu_stride;
            let gu_b_start = eidx * inter2;
            let dp_w_start = eidx * dp_stride;
            let dp_b_start = eidx * hidden_size;

            let input_t = Tensor::<B, 2>::from_data(TensorData::new(input_data, [n, hidden_size]), device);
            let gu_w_t = Tensor::<B, 2>::from_data(
                TensorData::new(self.gu_cache[gu_w_start..gu_w_start + gu_stride].to_vec(), [hidden_size, inter2]),
                device,
            );
            let gu_b_t = Tensor::<B, 1>::from_data(
                TensorData::new(self.gu_b_cache[gu_b_start..gu_b_start + inter2].to_vec(), [inter2]),
                device,
            );

            // gate_up = input @ gu_w + gu_b  — BLAS matmul
            let gate_up = input_t.matmul(gu_w_t) + gu_b_t.unsqueeze_dim(0);

            // Apply custom gating on CPU (element-wise, fast)
            let gate_up_data: Vec<f32> = gate_up.to_data().convert::<f32>().to_vec::<f32>().unwrap();
            let mut gated_data = vec![0f32; n * self.intermediate_size];
            for i in 0..n {
                let off = i * inter2;
                let g_off = i * self.intermediate_size;
                for j in 0..self.intermediate_size {
                    let gate = gate_up_data[off + j].min(LIMIT);
                    let up = gate_up_data[off + self.intermediate_size + j].clamp(-LIMIT, LIMIT);
                    let glu = gate * sigmoid(gate * ALPHA);
                    gated_data[g_off + j] = (up + 1.0) * glu;
                }
            }

            // down = gated @ dp_w + dp_b  — BLAS matmul
            let gated_t = Tensor::<B, 2>::from_data(TensorData::new(gated_data, [n, self.intermediate_size]), device);
            let dp_w_t = Tensor::<B, 2>::from_data(
                TensorData::new(self.dp_cache[dp_w_start..dp_w_start + dp_stride].to_vec(), [self.intermediate_size, hidden_size]),
                device,
            );
            let dp_b_t = Tensor::<B, 1>::from_data(
                TensorData::new(self.dp_b_cache[dp_b_start..dp_b_start + hidden_size].to_vec(), [hidden_size]),
                device,
            );

            let down = gated_t.matmul(dp_w_t) + dp_b_t.unsqueeze_dim(0);
            let down_data: Vec<f32> = down.to_data().convert::<f32>().to_vec::<f32>().unwrap();

            // Scatter-add with routing weights
            for (i, &(tidx, weight)) in assignments.iter().enumerate() {
                let src = i * hidden_size;
                let dst = tidx * hidden_size;
                for j in 0..hidden_size {
                    result_data[dst + j] += down_data[src + j] * weight;
                }
            }
        }

        // Scale by num_experts_per_tok and reshape back
        let result = Tensor::<B, 2>::from_data(
            TensorData::new(result_data, [total_tokens, hidden_size]),
            device,
        );
        result.mul_scalar(top_k as f32).reshape([batch, seq_len, hidden_size])
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

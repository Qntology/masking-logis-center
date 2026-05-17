/// Grouped Query Attention with sliding window, RoPE, and attention sinks.
///
/// Matches the Python OpenAIPrivacyFilterAttention exactly:
///   - Separate Q/K/V/O projections with bias
///   - GQA: 14 query heads, 2 KV heads (group size 7)
///   - Scaling: head_dim^(-0.25) applied separately to Q and K
///   - Attention sinks: per-head scalar appended to attention logits
///   - Max-subtract for numerical stability before softmax
///   - Softmax over [attn_weights | sinks], then drop sink column
///   - Bidirectional sliding window mask

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn;

use super::rope;

#[derive(Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: nn::Linear<B>,
    pub k_proj: nn::Linear<B>,
    pub v_proj: nn::Linear<B>,
    pub o_proj: nn::Linear<B>,
    pub sinks: Param<Tensor<B, 1>>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub scaling: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let q_proj = nn::LinearConfig::new(hidden_size, q_out)
            .with_bias(bias)
            .init(device);
        let k_proj = nn::LinearConfig::new(hidden_size, kv_out)
            .with_bias(bias)
            .init(device);
        let v_proj = nn::LinearConfig::new(hidden_size, kv_out)
            .with_bias(bias)
            .init(device);
        let o_proj = nn::LinearConfig::new(q_out, hidden_size)
            .with_bias(bias)
            .init(device);

        let sinks_tensor = Tensor::zeros([num_heads], device);
        let scaling = (head_dim as f32).powf(-0.25);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sinks: Param::initialized(ParamId::new(), sinks_tensor),
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups: num_heads / num_kv_heads,
            scaling,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `hidden_states`: [batch, seq_len, hidden_size]
    /// - `cos`: [1, seq_len, head_dim/2] from RoPE
    /// - `sin`: [1, seq_len, head_dim/2] from RoPE
    /// - `attention_mask`: [batch, 1, seq_len, seq_len] with 0 for allowed, -inf for masked
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: &Tensor<B, 3>,
        sin: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden_states.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape: [batch, seq_len, heads*head_dim] -> [batch, heads, seq_len, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE
        let (q, k) = rope::apply_rotary_emb(q, k, cos, sin);

        // Scale Q and K separately by head_dim^(-0.25)
        let q = q.mul_scalar(self.scaling);
        let k = k.mul_scalar(self.scaling);

        // Repeat KV heads for GQA
        let k = repeat_kv(k, self.num_kv_groups);
        let v = repeat_kv(v, self.num_kv_groups);

        // Attention weights: Q @ K^T
        // [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
        let attn_weights = q.matmul(k.swap_dims(2, 3));

        // Apply attention mask
        let attn_weights = attn_weights + attention_mask.clone();

        // Append sinks: [batch, heads, seq_len, 1]
        let sinks = self.sinks.val().clone()
            .reshape([1, self.num_heads, 1, 1])
            .expand([batch, self.num_heads, seq_len, 1]);
        let combined = Tensor::cat(vec![attn_weights, sinks], 3);
        // combined: [batch, heads, seq_len, seq_len + 1]

        // Max-subtract for numerical stability
        let max_vals = combined.clone().max_dim(3);
        let combined = combined - max_vals;

        // Softmax (in f32 — already f32 in Burn NdArray backend)
        let probs = burn::tensor::activation::softmax(combined, 3);

        // Drop the sink column (last element along dim 3)
        let scores = probs.slice([0..batch, 0..self.num_heads, 0..seq_len, 0..seq_len]);

        // Matmul with values
        let attn_output = scores.matmul(v);
        // [batch, heads, seq_len, head_dim]

        // Reshape back: [batch, seq_len, heads * head_dim]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(attn_output)
    }
}

/// Repeat KV heads to match query heads for GQA.
/// [batch, kv_heads, seq_len, head_dim] -> [batch, kv_heads * n_rep, seq_len, head_dim]
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, kv_heads, seq_len, head_dim] = x.dims();
    // [batch, kv_heads, 1, seq_len, head_dim]
    let x = x.unsqueeze_dim::<5>(2);
    // expand to [batch, kv_heads, n_rep, seq_len, head_dim]
    let x = x.expand([batch, kv_heads, n_rep, seq_len, head_dim]);
    // reshape to [batch, kv_heads * n_rep, seq_len, head_dim]
    x.reshape([batch, kv_heads * n_rep, seq_len, head_dim])
}

/// Create a bidirectional sliding window attention mask.
///
/// Returns a tensor of shape [1, 1, seq_len, seq_len] where:
///   mask[0, 0, i, j] = 0.0 if |i - j| <= window_size
///   mask[0, 0, i, j] = -1e9 otherwise
///
/// `window_size` is the config's sliding_window value (128), meaning
/// each token can attend to 128 tokens on each side plus itself.
pub fn create_sliding_window_mask<B: Backend>(
    seq_len: usize,
    window_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let n = seq_len;
    let mut mask_data = vec![0f32; n * n];
    let neg_inf: f32 = -1e9;

    for i in 0..n {
        for j in 0..n {
            let dist = if i > j { i - j } else { j - i };
            if dist > window_size {
                mask_data[i * n + j] = neg_inf;
            }
        }
    }

    Tensor::<B, 4>::from_data(
        TensorData::new(mask_data, [1, 1, n, n]),
        device,
    )
}

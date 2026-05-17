/// Full OpenAI Privacy Filter model for token classification.
///
/// Architecture:
///   1. Token embedding [vocab_size, hidden_size]
///   2. N transformer layers (attention + MoE)
///   3. Final RMSNorm
///   4. Classification head (Linear [hidden_size, num_labels])

use burn::prelude::*;
use burn::module::{Param, ParamId};

use super::layer::TransformerLayer;
use super::norm::RmsNorm;
use super::rope::RotaryEmbedding;
use super::attention::create_sliding_window_mask;
use crate::config::ModelConfig;

#[derive(Debug)]
pub struct PrivacyFilterModel<B: Backend> {
    pub embed_tokens: Param<Tensor<B, 2>>,  // [vocab_size, hidden_size]
    pub layers: Vec<TransformerLayer<B>>,
    pub norm: RmsNorm<B>,
    pub score_weight: Param<Tensor<B, 2>>,  // [hidden_size, num_labels] (transposed for burn)
    pub score_bias: Param<Tensor<B, 1>>,    // [num_labels]

    pub rope: RotaryEmbedding<B>,
    pub sliding_window: usize,

    pub hidden_size: usize,
    pub num_labels: usize,
}

impl<B: Backend> PrivacyFilterModel<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let num_labels = config.num_labels();

        // Token embedding
        let embed_tokens = Tensor::zeros([config.vocab_size, config.hidden_size], device);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(TransformerLayer::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.num_local_experts,
                config.num_experts_per_tok,
                config.rms_norm_eps,
                config.attention_bias,
                device,
            ));
        }

        // Final norm
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        // Classification head
        let score_weight = Tensor::zeros([config.hidden_size, num_labels], device);
        let score_bias = Tensor::zeros([num_labels], device);

        // RoPE
        let rp = &config.rope_parameters;
        let rope = RotaryEmbedding::new_yarn(
            config.head_dim,
            config.max_position_embeddings,
            rp.rope_theta,
            rp.factor,
            rp.beta_fast,
            rp.beta_slow,
            rp.original_max_position_embeddings,
            rp.truncate,
            device,
        );

        Self {
            embed_tokens: Param::initialized(ParamId::new(), embed_tokens),
            layers,
            norm,
            score_weight: Param::initialized(ParamId::new(), score_weight),
            score_bias: Param::initialized(ParamId::new(), score_bias),
            rope,
            sliding_window: config.sliding_window,
            hidden_size: config.hidden_size,
            num_labels,
        }
    }

    /// Run the full model forward pass.
    ///
    /// # Arguments
    /// - `input_ids`: [batch, seq_len] token IDs
    ///
    /// # Returns
    /// Logits tensor [batch, seq_len, num_labels]
    pub fn forward(
        &self,
        input_ids: &[u32],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let seq_len = input_ids.len();

        // 1. Token embedding via gather
        let ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let ids_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids_i64, [seq_len]),
            device,
        );
        let hidden_states = self.embed_tokens.val().clone().select(0, ids_tensor);
        // [seq_len, hidden_size]
        let hidden_states = hidden_states.unsqueeze_dim::<3>(0);
        // [1, seq_len, hidden_size]

        // 2. RoPE cos/sin
        let (cos, sin) = self.rope.get(seq_len);

        // 3. Sliding window attention mask
        let attention_mask = create_sliding_window_mask::<B>(seq_len, self.sliding_window, device);

        // 4. Transformer layers
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, &cos, &sin, &attention_mask, device);
        }

        // 5. Final norm
        hidden_states = self.norm.forward(hidden_states);

        // 6. Classification head: hidden_states @ score_weight + score_bias
        // hidden_states: [batch, seq_len, hidden_size]
        // score_weight: [hidden_size, num_labels]
        let logits = hidden_states.matmul(self.score_weight.val().clone().unsqueeze_dim::<3>(0))
            + self.score_bias.val().clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        // [batch, seq_len, num_labels]

        logits
    }
}

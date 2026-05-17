/// Transformer encoder layer combining attention and MoE.
///
/// Matches the Python OpenAIPrivacyFilterEncoderLayer:
///   residual = x
///   x = input_layernorm(x)
///   x = self_attn(x, ...)
///   x = residual + x
///   residual = x
///   x = post_attention_layernorm(x)
///   x = mlp(x)
///   x = residual + x

use burn::prelude::*;

use super::attention::Attention;
use super::moe::SparseMoE;
use super::norm::RmsNorm;

#[derive(Debug)]
pub struct TransformerLayer<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub self_attn: Attention<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub mlp: SparseMoE<B>,
}

impl<B: Backend> TransformerLayer<B> {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        rms_norm_eps: f64,
        attention_bias: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            input_layernorm: RmsNorm::new(hidden_size, rms_norm_eps, device),
            self_attn: Attention::new(hidden_size, num_heads, num_kv_heads, head_dim, attention_bias, device),
            post_attention_layernorm: RmsNorm::new(hidden_size, rms_norm_eps, device),
            mlp: SparseMoE::new(hidden_size, intermediate_size, num_experts, num_experts_per_tok, device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        cos: &Tensor<B, 3>,
        sin: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        // Self-attention block
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, cos, sin, attention_mask);
        let hidden_states = residual + hidden_states;

        // MoE block
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, device);
        residual + hidden_states
    }
}

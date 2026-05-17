/// RMSNorm matching the Python OpenAIPrivacyFilterRMSNorm exactly:
///
///   hidden_states = hidden_states.to(float32)
///   variance = hidden_states.pow(2).mean(-1, keepdim=True)
///   hidden_states = hidden_states * rsqrt(variance + eps)
///   return (weight * hidden_states).to(input_dtype)

use burn::prelude::*;
use burn::module::{Param, ParamId};

#[derive(Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::ones([size], device);
        Self {
            weight: Param::initialized(ParamId::new(), weight),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // variance = x.pow(2).mean(-1, keepdim=True)
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        // rsqrt(variance + eps)
        let scale = (variance + self.eps).sqrt().recip();
        // weight * (x * scale)
        let normed = x * scale;
        normed * self.weight.val().clone().unsqueeze::<3>()
    }
}

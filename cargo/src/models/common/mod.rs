pub mod modules;
pub mod generate;
pub mod gguf;
pub mod blocks;

pub use blocks::*;

use anyhow::Result;
use candle_core::Tensor;

pub trait InferenceModel {
    fn forward_initial(&mut self, input_ids: &Tensor, seqlen_offset: usize, data: MultiModalData) -> Result<Tensor>;
    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;
    fn clear_cache(&mut self);
    fn stop_token_ids(&self) -> Vec<u32>;
}

pub struct MultiModalData {
    pub data_vec: Vec<Tensor>,
}

impl MultiModalData {
    pub fn new(data_vec: Vec<Tensor>) -> Self {
        Self { data_vec }
    }
}

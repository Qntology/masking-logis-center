//! # privacy-filter-rs — OpenAI Privacy Filter inference in Rust
//!
//! Pure-Rust inference for the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)
//! token classification model, built on [Burn 0.20](https://burn.dev).
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use privacy_filter_rs::PrivacyFilterInference;
//! use std::path::Path;
//!
//! // Choose backend
//! type B = burn::backend::NdArray;
//! let device = burn::backend::ndarray::NdArrayDevice::Cpu;
//!
//! let engine = PrivacyFilterInference::<B>::load(
//!     Path::new("/path/to/privacy-filter"),
//!     device,
//! )?;
//!
//! let spans = engine.predict("My name is Alice Smith")?;
//! for span in &spans {
//!     println!("{}: {} (score: {:.4})", span.entity_group, span.word, span.score);
//! }
//! ```

// ── Thread configuration ────────────────────────────────────────────────────

/// Configure the global Rayon thread pool.
///
/// Call this **once**, before any model operations.
/// Returns the actual number of threads in the pool.
pub fn init_threads(n: Option<usize>) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(count) = n {
        if count > 0 {
            builder = builder.num_threads(count);
        }
    }
    let _ = builder.build_global();
    rayon::current_num_threads()
}

// ── Internal modules ────────────────────────────────────────────────────────

pub mod config;
pub mod inference;
pub mod model;
pub mod tensor_utils;
pub mod viterbi;
pub mod weights;

// ── Flat re-exports ─────────────────────────────────────────────────────────

pub use config::{ModelConfig, ViterbiConfig};
pub use inference::PrivacyFilterInference;
pub use viterbi::PrivacySpan;

// ── Backend selection ───────────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
pub mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
}

#[cfg(all(feature = "wgpu-f16", not(feature = "ndarray"), not(feature = "wgpu"), not(feature = "mlx")))]
pub mod backend {
    pub type B = burn::backend::wgpu::Wgpu<half::f16, i32, u32>;
    pub type Device = burn::backend::wgpu::WgpuDevice;
    pub fn device() -> Device { Device::DefaultDevice }
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray"), not(feature = "wgpu-f16"), not(feature = "mlx")))]
pub mod backend {
    pub use burn::backend::Wgpu as B;
    pub type Device = burn::backend::wgpu::WgpuDevice;
    pub fn device() -> Device { Device::DefaultDevice }
}

#[cfg(all(feature = "mlx", not(feature = "ndarray"), not(feature = "wgpu"), not(feature = "wgpu-f16")))]
pub mod backend {
    pub use burn_mlx::Mlx as B;
    pub type Device = burn_mlx::MlxDevice;
    pub fn device() -> Device { Default::default() }
}

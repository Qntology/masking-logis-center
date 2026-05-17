/// YaRN Rotary Position Embeddings with **interleaved** layout.
///
/// Matches the Python OpenAIPrivacyFilterRotaryEmbedding exactly:
///   - YaRN interpolation of inv_freq
///   - Interleaved application: first_half = x[..., ::2], second_half = x[..., 1::2]
///   - cos/sin scaled by attention_factor

use burn::prelude::*;

/// Precomputed cos/sin tables for RoPE.
#[derive(Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// [max_seq_len, head_dim/2]
    pub cos: Tensor<B, 2>,
    /// [max_seq_len, head_dim/2]
    pub sin: Tensor<B, 2>,
    pub attention_scaling: f64,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Build YaRN RoPE tables.
    ///
    /// # Parameters
    /// - `head_dim`: dimension per head (64)
    /// - `max_seq_len`: max positions to precompute
    /// - `rope_theta`: base frequency (150000.0)
    /// - `factor`: YaRN scaling factor (32.0)
    /// - `beta_fast`: YaRN fast boundary (32.0)
    /// - `beta_slow`: YaRN slow boundary (1.0)
    /// - `original_max_pos`: original max position embeddings (4096)
    /// - `truncate`: whether to floor/ceil correction range bounds
    pub fn new_yarn(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        original_max_pos: usize,
        truncate: bool,
        device: &B::Device,
    ) -> Self {
        let dim = head_dim;
        let half_dim = dim / 2;

        // Compute attention_factor = get_mscale(factor)
        let attention_scaling = if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        };

        // pos_freqs = base^(2i/dim) for i in 0..half_dim
        let mut inv_freq_extrapolation = vec![0f32; half_dim];
        let mut inv_freq_interpolation = vec![0f32; half_dim];
        for i in 0..half_dim {
            let freq = rope_theta.powf(2.0 * i as f64 / dim as f64);
            inv_freq_extrapolation[i] = 1.0 / freq as f32;
            inv_freq_interpolation[i] = 1.0 / (factor * freq) as f32;
        }

        // find_correction_range
        let find_correction_dim = |num_rotations: f64| -> f64 {
            (dim as f64
                * (original_max_pos as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
                / (2.0 * rope_theta.ln())
        };

        let low_raw = find_correction_dim(beta_fast);
        let high_raw = find_correction_dim(beta_slow);

        let (low, high) = if truncate {
            (low_raw.floor(), high_raw.ceil())
        } else {
            (low_raw, high_raw)
        };
        let low = low.max(0.0);
        let high = high.min((dim - 1) as f64);

        // linear_ramp_factor
        let max_val = if (high - low).abs() < 1e-9 {
            high + 0.001
        } else {
            high
        };
        let mut ramp = vec![0f32; half_dim];
        for i in 0..half_dim {
            let linear = (i as f64 - low) / (max_val - low);
            ramp[i] = linear.clamp(0.0, 1.0) as f32;
        }

        // inv_freq = interpolation * ramp + extrapolation * (1 - ramp)
        let mut inv_freq = vec![0f32; half_dim];
        for i in 0..half_dim {
            inv_freq[i] = inv_freq_interpolation[i] * ramp[i]
                + inv_freq_extrapolation[i] * (1.0 - ramp[i]);
        }

        // Build cos/sin tables: for each position p, compute p * inv_freq[i]
        let mut cos_data = vec![0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0f32; max_seq_len * half_dim];
        let scale = attention_scaling as f32;

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                cos_data[pos * half_dim + i] = angle.cos() * scale;
                sin_data[pos * half_dim + i] = angle.sin() * scale;
            }
        }

        let cos = Tensor::<B, 2>::from_data(
            TensorData::new(cos_data, [max_seq_len, half_dim]),
            device,
        );
        let sin = Tensor::<B, 2>::from_data(
            TensorData::new(sin_data, [max_seq_len, half_dim]),
            device,
        );

        Self {
            cos,
            sin,
            attention_scaling,
        }
    }

    /// Get cos/sin for the given sequence length.
    /// Returns (cos, sin) each of shape [1, seq_len, half_dim].
    pub fn get(&self, seq_len: usize) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let cos = self.cos.clone().slice([0..seq_len]).unsqueeze_dim::<3>(0);
        let sin = self.sin.clone().slice([0..seq_len]).unsqueeze_dim::<3>(0);
        (cos, sin)
    }
}

/// Apply RoPE with interleaved layout to Q and K tensors.
///
/// Input shapes:
///   q: [batch, num_heads, seq_len, head_dim]
///   k: [batch, num_kv_heads, seq_len, head_dim]
///   cos: [1, seq_len, head_dim/2]
///   sin: [1, seq_len, head_dim/2]
///
/// The interleaved layout means:
///   first_half = x[..., ::2]  (even indices)
///   second_half = x[..., 1::2]  (odd indices)
///   rotated_first = first_half * cos - second_half * sin
///   rotated_second = second_half * cos + first_half * sin
///   result = interleave(rotated_first, rotated_second)
pub fn apply_rotary_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: &Tensor<B, 3>,
    sin: &Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let q_rot = apply_rotary_emb_single(q, cos, sin);
    let k_rot = apply_rotary_emb_single(k, cos, sin);
    (q_rot, k_rot)
}

fn apply_rotary_emb_single<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 3>,
    sin: &Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, heads, seq_len, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    // Extract interleaved halves: x[..., ::2] and x[..., 1::2]
    // Reshape to [batch, heads, seq_len, half_dim, 2] then slice
    let x_pairs = x.reshape([batch, heads, seq_len, half_dim, 2]);
    let first_half = x_pairs.clone().slice([0..batch, 0..heads, 0..seq_len, 0..half_dim, 0..1])
        .reshape([batch, heads, seq_len, half_dim]);
    let second_half = x_pairs.slice([0..batch, 0..heads, 0..seq_len, 0..half_dim, 1..2])
        .reshape([batch, heads, seq_len, half_dim]);

    // cos/sin: [1, seq_len, half_dim] -> [1, 1, seq_len, half_dim] for broadcasting
    let cos = cos.clone().unsqueeze_dim::<4>(1);
    let sin = sin.clone().unsqueeze_dim::<4>(1);

    // Apply rotation
    let rotated_first = first_half.clone() * cos.clone() - second_half.clone() * sin.clone();
    let rotated_second = second_half * cos + first_half * sin;

    // Interleave back: stack on last dim then flatten
    // [batch, heads, seq_len, half_dim] -> [batch, heads, seq_len, half_dim, 1]
    let rf = rotated_first.unsqueeze_dim::<5>(4);
    let rs = rotated_second.unsqueeze_dim::<5>(4);

    // Cat along dim 4: [batch, heads, seq_len, half_dim, 2]
    let interleaved = Tensor::cat(vec![rf, rs], 4);

    // Flatten last two dims: [batch, heads, seq_len, head_dim]
    interleaved.reshape([batch, heads, seq_len, head_dim])
}

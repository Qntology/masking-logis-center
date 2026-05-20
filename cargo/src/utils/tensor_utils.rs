use candle_core::{Device, Result, Tensor, DType};

pub fn repeat_kv(x: Tensor, num_repeats: usize) -> Result<Tensor> {
    if num_repeats == 1 {
        return Ok(x);
    }
    let (b, n_kv, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, num_repeats, l, d))?
        .flatten(1, 2)
}

pub fn prepare_causal_attention_mask(
    _bs: usize,
    seq_len: usize,
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len + seqlen_offset).map(move |j| {
                if j > i + seqlen_offset {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len + seqlen_offset), device)
}

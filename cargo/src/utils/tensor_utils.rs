use candle_core::{Device, Result, Tensor}; // DType 제거

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
    bs: usize,
    seq_len: usize,
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len + seqlen_offset).map(move |j| {
                if j > i + seqlen_offset {
                    -1e9f32 // 🚀 f32::NEG_INFINITY는 BF16 변환 시 NaN을 유발하여 무한 반복(Hallucination)에 빠지는 핵심 원인이 되므로 안전한 최소값으로 대체합니다.
                } else {
                    0f32
                }
            })
        })
        .collect();
    let mask_2d = Tensor::from_slice(&mask, (seq_len, seq_len + seqlen_offset), device)?;
    
    // 🚀 Attention 연산의 4D 차원(Batch, Heads, Q_len, K_len)에 맞춰 브로드캐스팅 및 narrow가 가능하도록
    // 1차원과 2차원에 unsqueeze를 적용하여 [bs, 1, seq_len, seq_len + offset] 형태로 확장합니다.
    mask_2d.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((bs, 1, seq_len, seq_len + seqlen_offset))
}

use candle_core::{Result, Tensor};

pub fn apply_rotary_pos_emb_vision(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q = apply_rope(q, cos, sin)?;
    let k = apply_rope(k, cos, sin)?;
    Ok((q, k))
}

pub fn glm_ocr_apply_rotary_pos_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q = apply_rope(q, cos, sin)?;
    let k = apply_rope(k, cos, sin)?;
    Ok((q, k))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
    
    // Broadcast cos and sin
    let cos = cos.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?;
    
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

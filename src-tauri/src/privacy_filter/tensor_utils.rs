/// Convert TensorData bytes to Vec<f32>, handling f32, f16, and bf16 element types.
///
/// Works around burn's `.to_vec::<f32>()` failing with TypeMismatch on
/// backends that use half-precision floats (e.g. Wgpu<half::f16, ..>).
pub fn tensor_data_to_f32(data: burn::tensor::TensorData) -> Vec<f32> {
    // Try direct f32 extraction first.
    if let Ok(v) = data.to_vec::<f32>() {
        return v;
    }
    // Try burn's convert path.
    let converted = data.clone().convert::<f32>();
    if let Ok(v) = converted.to_vec::<f32>() {
        return v;
    }
    // Manual f16→f32 conversion from raw bytes.
    let bytes = &data.bytes;
    if bytes.len() % 2 == 0 {
        return bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
    }
    panic!("cannot convert tensor data ({} bytes) to f32", bytes.len());
}

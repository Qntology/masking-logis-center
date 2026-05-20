use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_nn::{Embedding, Module};

pub struct QLinear {
    inner: QMatMul,
}

impl QLinear {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let tensor = ct.tensor(reader, name, device)?;
        let qmm = QMatMul::from_qtensor(tensor)?;
        Ok(Self { inner: qmm })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let shape = x.shape();
        let dims = shape.dims();
        let last_dim = dims[dims.len() - 1];
        let leading_dims: usize = dims[..dims.len() - 1].iter().product();
        
        let x_flat = x.reshape((leading_dims, last_dim))?;
        let out = self.inner.forward(&x_flat)?;
        
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(out.dim(1)?);
        let out = out.reshape(out_dims)?;
        out.to_dtype(x_dtype)
    }
}

pub struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
        eps: f64,
    ) -> Result<Self> {
        let weight = ct.tensor(reader, name, device)?.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

pub struct QEmbedding {
    inner: Embedding,
}

impl QEmbedding {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let tensor = ct.tensor(reader, name, device)?.dequantize(device)?;
        let _vocab_size = tensor.dim(0)?;
        let hidden_size = tensor.dim(1)?;
        Ok(Self {
            inner: Embedding::new(tensor, hidden_size),
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

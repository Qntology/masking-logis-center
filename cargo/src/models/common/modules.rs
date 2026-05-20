use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Activation, Linear, Module, VarBuilder};

pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        intermediate_dim: usize,
        act: Activation,
        _bias: bool,
        gate_name: Option<&str>,
        up_name: Option<&str>,
        down_name: Option<&str>,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(in_dim, intermediate_dim, vb.pp(gate_name.unwrap_or("gate_proj")))?;
        let up_proj = linear_no_bias(in_dim, intermediate_dim, vb.pp(up_name.unwrap_or("up_proj")))?;
        let down_proj = linear_no_bias(intermediate_dim, in_dim, vb.pp(down_name.unwrap_or("down_proj")))?;
        Ok(Self { gate_proj, up_proj, down_proj, act })
    }

    pub fn new_dummy(device: &candle_core::Device) -> Self {
        let dummy = Tensor::zeros((1, 1), candle_core::DType::F32, device).unwrap();
        Self {
            gate_proj: Linear::new(dummy.clone(), None),
            up_proj: Linear::new(dummy.clone(), None),
            down_proj: Linear::new(dummy, None),
            act: Activation::Silu,
        }
    }
    
    pub fn clear_weights(&mut self) {
        let dummy = Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        self.gate_proj = Linear::new(dummy.clone(), None);
        self.up_proj = Linear::new(dummy.clone(), None);
        self.down_proj = Linear::new(dummy, None);
    }
    
    pub fn load_weights_inplace<R: std::io::Read + std::io::Seek>(&mut self, ct: &candle_core::quantized::gguf_file::Content, reader: &mut R, prefix: &str, device: &candle_core::Device, dtype: candle_core::DType, bias: bool) -> Result<()> {
        // reader를 변경하는 클로저이므로 mut 선언이 필수입니다.
        let mut get_lin = |name: &str| -> Result<Linear> {
            let w = ct.tensor(reader, &format!("{}.weight", name), device)?;
            let w = w.dequantize_f16(device).or_else(|_| w.dequantize(device))?.to_dtype(dtype)?;
            let b = if bias {
                if let Ok(b_t) = ct.tensor(reader, &format!("{}.bias", name), device) {
                    Some(b_t.dequantize_f16(device).or_else(|_| b_t.dequantize(device))?.to_dtype(dtype)?)
                } else { None }
            } else { None };
            Ok(Linear::new(w, b))
        };
        self.gate_proj = get_lin(&format!("{}ffn_gate", prefix))?;
        self.up_proj = get_lin(&format!("{}ffn_up", prefix))?;
        self.down_proj = get_lin(&format!("{}ffn_down", prefix))?;
        Ok(())
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let act = self.act.forward(&gate)?;
        self.down_proj.forward(&(act * up)?)
    }
}

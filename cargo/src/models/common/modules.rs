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
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let act = self.act.forward(&gate)?;
        self.down_proj.forward(&(act * up)?)
    }
}

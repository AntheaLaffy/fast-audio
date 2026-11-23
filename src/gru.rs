use crate::{
    Result, Shape, Tensor,
    cpu::Cpu,
    error::{int, invalid},
};

/// One-layer, batch-first, reset-after GRU. Gate order is reset, update, new.
#[derive(Clone, Copy, Debug)]
pub struct GruSpec {
    pub batch: usize,
    pub steps: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub bidirectional: bool,
}
impl GruSpec {
    pub fn directions(&self) -> usize {
        if self.bidirectional { 2 } else { 1 }
    }
    pub fn input_shape(&self) -> [usize; 3] {
        [self.batch, self.steps, self.input_size]
    }
    pub fn state_shape(&self) -> [usize; 3] {
        [self.directions(), self.batch, self.hidden_size]
    }
    pub fn output_shape(&self) -> Result<[usize; 3]> {
        if [self.batch, self.steps, self.input_size, self.hidden_size].contains(&0) {
            return Err(invalid("GRU dimensions must be positive"));
        }
        for value in [self.batch, self.steps, self.input_size, self.hidden_size] {
            int(value)?;
        }
        let gates = self
            .hidden_size
            .checked_mul(3)
            .ok_or_else(|| invalid("GRU gate dimension overflows"))?;
        let rows = self
            .batch
            .checked_mul(self.steps)
            .ok_or_else(|| invalid("GRU sequence size overflows"))?;
        int(gates)?;
        int(rows)?;
        let output = [self.batch, self.steps, self.directions() * self.hidden_size];
        for dims in [
            self.input_shape(),
            self.state_shape(),
            output,
            [self.directions(), gates, self.input_size],
            [self.directions(), gates, self.hidden_size],
            [self.batch, self.steps, gates],
        ] {
            Shape::new(dims)?;
        }
        int(self.batch * self.hidden_size)?;
        Ok(output)
    }
    pub(crate) fn check_weights(&self, weights: &GruWeights) -> Result<()> {
        self.output_shape()?;
        let d = self.directions();
        let gates = self.hidden_size * 3;
        weights.input.shape().expect(&[d, gates, self.input_size])?;
        weights
            .hidden
            .shape()
            .expect(&[d, gates, self.hidden_size])?;
        weights.input_bias.shape().expect(&[d, gates])?;
        weights.hidden_bias.shape().expect(&[d, gates])
    }
}

/// Parameters are stacked by direction, then by reset/update/new gate.
/// Plans snapshot these tensors; later source changes do not alter a prepared GRU.
pub struct GruWeights {
    pub input: Tensor,
    pub hidden: Tensor,
    pub input_bias: Tensor,
    pub hidden_bias: Tensor,
}

/// Prepared CPU GRU with batched input projection and reusable recurrent storage.
pub struct Gru {
    cpu: Cpu,
    spec: GruSpec,
    input_weights: Vec<f32>,
    hidden_weights: Vec<f32>,
    input_bias: Vec<f32>,
    hidden_bias: Vec<f32>,
    projected: Vec<f32>,
    recurrent: Vec<f32>,
    state: Vec<f32>,
}
impl Gru {
    pub fn new(cpu: Cpu, spec: GruSpec, weights: &GruWeights) -> Result<Self> {
        spec.check_weights(weights)?;
        let d = spec.directions();
        let gates = spec.hidden_size * 3;
        fn transpose(input: &[f32], d: usize, rows: usize, columns: usize) -> Vec<f32> {
            let mut out = vec![0.0; input.len()];
            for direction in 0..d {
                for row in 0..rows {
                    for column in 0..columns {
                        out[(direction * columns + column) * rows + row] =
                            input[(direction * rows + row) * columns + column];
                    }
                }
            }
            out
        }
        Ok(Self {
            cpu,
            spec,
            input_weights: transpose(weights.input.data(), d, gates, spec.input_size),
            hidden_weights: transpose(weights.hidden.data(), d, gates, spec.hidden_size),
            input_bias: weights.input_bias.data().to_vec(),
            hidden_bias: weights.hidden_bias.data().to_vec(),
            projected: vec![0.0; spec.batch * spec.steps * gates],
            recurrent: vec![0.0; spec.batch * gates],
            state: vec![0.0; spec.batch * spec.hidden_size],
        })
    }
    pub fn spec(&self) -> GruSpec {
        self.spec
    }
    pub fn workspace_bytes(&self) -> usize {
        (self.projected.len() + self.recurrent.len() + self.state.len()) * size_of::<f32>()
    }
    pub fn parameter_bytes(&self) -> usize {
        (self.input_weights.len()
            + self.hidden_weights.len()
            + self.input_bias.len()
            + self.hidden_bias.len())
            * size_of::<f32>()
    }
    pub fn run(
        &mut self,
        input: &Tensor,
        initial: &Tensor,
        output: &mut Tensor,
        final_state: &mut Tensor,
    ) -> Result<()> {
        let s = self.spec;
        let h = s.hidden_size;
        let g = 3 * h;
        let d = s.directions();
        input.shape().expect(&s.input_shape())?;
        initial.shape().expect(&s.state_shape())?;
        output.shape().expect(&[s.batch, s.steps, d * h])?;
        final_state.shape().expect(&s.state_shape())?;
        for direction in 0..d {
            self.cpu.gemm(
                s.batch * s.steps,
                g,
                s.input_size,
                input.data(),
                &self.input_weights[direction * s.input_size * g..][..s.input_size * g],
                &mut self.projected,
            );
            self.state
                .copy_from_slice(&initial.data()[direction * s.batch * h..][..s.batch * h]);
            let bi = &self.input_bias[direction * g..][..g];
            let bh = &self.hidden_bias[direction * g..][..g];
            for index in 0..s.steps {
                let time = if direction == 0 {
                    index
                } else {
                    s.steps - 1 - index
                };
                self.cpu.gemm(
                    s.batch,
                    g,
                    h,
                    &self.state,
                    &self.hidden_weights[direction * h * g..][..h * g],
                    &mut self.recurrent,
                );
                for batch in 0..s.batch {
                    for unit in 0..h {
                        let u = &self.projected[(batch * s.steps + time) * g..][..g];
                        let v = &self.recurrent[batch * g..][..g];
                        let reset =
                            1.0 / (1.0 + (-(u[unit] + bi[unit] + v[unit] + bh[unit])).exp());
                        let update = 1.0
                            / (1.0
                                + (-(u[h + unit] + bi[h + unit] + v[h + unit] + bh[h + unit]))
                                    .exp());
                        let new = (u[2 * h + unit]
                            + bi[2 * h + unit]
                            + reset * (v[2 * h + unit] + bh[2 * h + unit]))
                            .tanh();
                        let value = (1.0 - update) * new + update * self.state[batch * h + unit];
                        self.state[batch * h + unit] = value;
                        output.data_mut()[((batch * s.steps + time) * d + direction) * h + unit] =
                            value;
                    }
                }
            }
            final_state.data_mut()[direction * s.batch * h..][..s.batch * h]
                .copy_from_slice(&self.state);
        }
        Ok(())
    }
}

use super::Operation;
use crate::{GradientMode, Layout, Result, Shape, Trackable};

pub(crate) type Read<'a, T> = (&'a T, &'a Layout);
pub(crate) type Write<'a, T> = (&'a mut T, &'a Layout);
#[derive(Default, Clone, Copy)]
pub(crate) struct PlanMemory {
    pub workspace: usize,
    pub metadata: usize,
    pub indices: usize,
    pub shared: usize,
}
pub(crate) trait Backend {
    type Tensor: Trackable;
    type Forward;
    type Backward;
    type Projection;
    fn check(&self, tensor: &Self::Tensor) -> Result<()>;
    fn zeros(&self, shape: &Shape) -> Result<Self::Tensor>;
    fn scalar(&self, value: f32) -> Result<Self::Tensor>;
    fn clear(&self, tensor: &mut Self::Tensor) -> Result<()>;
    fn update(
        &self,
        parameter: &mut Self::Tensor,
        gradient: &Self::Tensor,
        rate: f32,
    ) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
    fn forward_plan(
        &self,
        op: &Operation,
        inputs: [Option<&Layout>; 3],
        output: &Layout,
    ) -> Result<Self::Forward>;
    fn backward_plan(
        &self,
        op: &Operation,
        operand: usize,
        inputs: [Option<&Layout>; 3],
        upstream: &Layout,
        output: &Layout,
    ) -> Result<Self::Backward>;
    fn dependencies(plan: &Self::Backward) -> [bool; 3];
    fn forward_memory(plan: &Self::Forward) -> PlanMemory;
    fn backward_memory(plan: &Self::Backward) -> PlanMemory;
    fn project_memory(plan: &Self::Projection) -> PlanMemory;
    fn project_plan(
        &self,
        base: &Shape,
        view: &Layout,
        upstream: &Layout,
        limit: usize,
    ) -> Result<Self::Projection>;
    fn forward(
        &self,
        plan: &mut Self::Forward,
        inputs: [Option<Read<'_, Self::Tensor>>; 3],
        output: Write<'_, Self::Tensor>,
    ) -> Result<()>;
    fn backward(
        &self,
        plan: &mut Self::Backward,
        inputs: [Option<Read<'_, Self::Tensor>>; 3],
        upstream: Read<'_, Self::Tensor>,
        output: Write<'_, Self::Tensor>,
        mode: GradientMode,
    ) -> Result<()>;
    fn project(
        &self,
        plan: &Self::Projection,
        upstream: Read<'_, Self::Tensor>,
        output: &mut Self::Tensor,
        mode: GradientMode,
    ) -> Result<()>;
}

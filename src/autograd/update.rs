use super::{
    GradientStatus, Value,
    backend::Backend,
    engine::{Core, check_version, read_value},
};
use crate::{Result, error::invalid};

impl<B: Backend> Core<B> {
    pub(crate) fn sgd(&mut self, leaves: &[Value], rate: f32) -> Result<()> {
        if !rate.is_finite() || rate < 0. {
            return Err(invalid("SGD rate must be finite and nonnegative"));
        }
        if leaves.is_empty() {
            return Ok(());
        }
        if self.gradient_state != GradientStatus::Valid || !self.forward_ready || !self.update_ready
        {
            return Err(invalid(
                "SGD requires valid gradients from backward on the current forward",
            ));
        }
        self.selected.fill(false);
        for &leaf in leaves {
            let slot = self
                .leaf_gradient(leaf)?
                .ok_or_else(|| invalid("SGD selected an unused or non-gradient leaf"))?;
            self.selected[slot] = true;
        }
        for (selected, gradient) in self.selected.iter().zip(&self.gradients) {
            if !selected {
                continue;
            }
            let root = gradient.owner_root;
            {
                let parameter =
                    read_value(&self.backend, &self.owners, &self.program.0.values, root)?;
                check_version(parameter.version(), self.forward_versions[root].unwrap())?;
            }
            self.owners[root].as_ref().unwrap().check_writable()?;
        }
        self.forward_ready = false;
        self.update_ready = false;
        self.gradient_state = GradientStatus::Invalid;
        for (selected, gradient) in self.selected.iter().zip(&self.gradients) {
            if !selected {
                continue;
            }
            let root = gradient.owner_root;
            // Recheck under the actual writer: a preflight lease cannot exclude
            // concurrent aliases after it is released. Prior writes are not rolled back.
            let mut parameter = self.owners[root]
                .as_ref()
                .unwrap()
                .write_checked(self.forward_versions[root].unwrap())?;
            self.backend.update(&mut parameter, &gradient.data, rate)?;
        }
        self.gradient_state = GradientStatus::Valid;
        Ok(())
    }
}

use crate::izikevich_model::{IzikevichModel, IzikevichModelState};
use nalgebra::DVector;

impl IzikevichModel {
    pub fn euler_step(&self, model_state: &mut IzikevichModelState, input_current: &DVector<f64>) {
        let firing_neurons = model_state.reset_firing_neurons(self.c, self.d, self.spike_trashhold);
        let adjusted_current = input_current
            + firing_neurons
                .iter()
                .fold(DVector::zeros(self.number_of_neurons), |acc, e| {
                    acc + self.connectivity_matrix.column(*e)
                });
        model_state.adjust_state(self.dt, adjusted_current, self.a, self.b);
    }
}

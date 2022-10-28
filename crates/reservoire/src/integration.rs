use std::iter::repeat_with;

use crate::izikevich_model::{IzhikevichModel, IzhikevichModelState, ThalmicInput};
use nalgebra::DVector;
use rand_distr::{Distribution, Normal};

impl IzhikevichModel {
    fn thalmic_input(&self) -> DVector<f64> {
        match self.thalmic_input {
            ThalmicInput::Const(mean) => DVector::zeros(self.number_of_neurons).add_scalar(mean),
            ThalmicInput::Normal { mean, dev } => {
                let mut normal_rng = rand::thread_rng();
                let normal_distribution = Normal::new(mean, dev).unwrap();
                let normal_iter = repeat_with(|| normal_distribution.sample(&mut normal_rng))
                    .take(self.number_of_neurons);
                DVector::from_iterator(self.number_of_neurons, normal_iter)
            }
        }
    }

    pub fn euler_step(&self, model_state: &mut IzhikevichModelState, input_current: &DVector<f64>) {
        let thalmic_input = self.thalmic_input();
        let firing_neurons = model_state.reset_firing_neurons(self.c, self.d, self.spike_trashhold);
        let adjusted_current = input_current
            + thalmic_input
            + firing_neurons
                .iter()
                .fold(DVector::zeros(self.number_of_neurons), |acc, e| {
                    acc + self.connectivity_matrix.column(*e)
                });
        model_state.adjust_state(self.dt, adjusted_current, self.a, self.b);
    }
}

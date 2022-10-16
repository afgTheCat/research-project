use crate::izikevich_model::{IzikevichModel, IzikevichModelState};
use nalgebra::{DMatrix, DVector};
use ode_solvers::System;

#[derive(Debug, Clone)]
pub struct ModelIntegrator {
    a: f64,
    b: f64,
    number_of_neurons: usize,
    spike_value: f64,
    current_input: DVector<f64>,
    connectivity_matrix: DMatrix<f64>,
}

impl ModelIntegrator {
    fn current_input(&self) -> &DVector<f64> {
        &self.current_input
    }

    pub fn new(
        izikevich_model: &IzikevichModel,
        current_input: &DVector<f64>,
        connectivity_matrix: &DMatrix<f64>,
    ) -> Self {
        Self {
            a: izikevich_model.a,
            b: izikevich_model.b,
            number_of_neurons: izikevich_model.number_of_neurons,
            spike_value: izikevich_model.spike_value,
            current_input: current_input.clone(),
            connectivity_matrix: connectivity_matrix.clone(),
        }
    }
}

impl System<IzikevichModelState> for ModelIntegrator {
    fn system(&self, time: f64, y: &IzikevichModelState, dy: &mut IzikevichModelState) {
        let current = self.current_input();

        let v_slice = y.slice((0, 0), (self.number_of_neurons, 1));
        let u_slice = y.slice((self.number_of_neurons, 0), (self.number_of_neurons, 1));
        let new_v_slice: DVector<f64> =
            ((0.04 * v_slice.component_mul(&v_slice)) + (v_slice * 5.0)).add_scalar(140.0)
                - u_slice
                + current;
        let new_u_slice = self.a * (self.b * v_slice - u_slice);
        let mut dv_slice = dy.slice_mut((0, 0), (self.number_of_neurons, 1));
        dv_slice.set_column(0, &new_v_slice);
        let mut du_slice = dy.slice_mut((self.number_of_neurons, 0), (self.number_of_neurons, 1));
        du_slice.set_column(0, &new_u_slice.column(0));

        for i in 0..self.number_of_neurons {
            let v_i = y[i];
            let w = self
                .connectivity_matrix
                .slice((i, 0), (1, self.number_of_neurons))
                * v_slice.add_scalar(-v_i);
            dy[i] += w[(0, 0)];
        }
    }

    fn solout(&mut self, _time: f64, y: &IzikevichModelState, _dy: &IzikevichModelState) -> bool {
        y.iter().any(|x| x > &self.spike_value)
    }
}

use crate::izikevich_model::{IzikevichModel, IzikevichModelState, IzikevichModelStateChemical};
use nalgebra::{DMatrix, DVector};
// use ode_solvers::System;

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
            spike_value: izikevich_model.spike_trashhold,
            current_input: current_input.clone(),
            connectivity_matrix: connectivity_matrix.clone(),
        }
    }
}

impl ModelIntegrator {
    fn system(
        &self,
        time: f64,
        y: &IzikevichModelStateChemical,
        dy: &mut IzikevichModelStateChemical,
    ) {
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

    fn solout(
        &mut self,
        _time: f64,
        y: &IzikevichModelStateChemical,
        _dy: &IzikevichModelStateChemical,
    ) -> bool {
        y.iter().any(|x| x > &self.spike_value)
    }
}

// integrate manually using euler's method
pub fn euler_integrate_step(
    model_integrator: &ModelIntegrator,
    model_state: &mut IzikevichModelStateChemical,
    dt: f64,
    c: f64,
    d: f64,
) {
    let mut dmodel_state = DVector::<f64>::zeros(2 * model_integrator.number_of_neurons);
    let input_current = model_integrator.current_input();
    let v_slice = model_state.slice((0, 0), (model_integrator.number_of_neurons, 1));
    let u_slice = model_state.slice(
        (model_integrator.number_of_neurons, 0),
        (model_integrator.number_of_neurons, 1),
    );
    let new_v_slice: DVector<f64> =
        ((0.04 * v_slice.component_mul(&v_slice)) + (v_slice * 5.0)).add_scalar(140.0) - u_slice
            + input_current;
    let new_u_slice = model_integrator.a * (model_integrator.b * v_slice - u_slice);
    let mut dv_slice = dmodel_state.slice_mut((0, 0), (model_integrator.number_of_neurons, 1));
    dv_slice.set_column(0, &new_v_slice);
    let mut du_slice = dmodel_state.slice_mut(
        (model_integrator.number_of_neurons, 0),
        (model_integrator.number_of_neurons, 1),
    );
    du_slice.set_column(0, &new_u_slice.column(0));

    for i in 0..model_integrator.number_of_neurons {
        let v_i = model_state[i];
        let w = model_integrator
            .connectivity_matrix
            .slice((i, 0), (1, model_integrator.number_of_neurons))
            * v_slice.add_scalar(-v_i);
        dmodel_state[i] += w[(0, 0)];
    }

    dmodel_state.scale_mut(dt);
    *model_state += dmodel_state;

    for n_index in 0..model_integrator.number_of_neurons {
        if model_state[n_index] > model_integrator.spike_value {
            model_state[n_index] = c;
            model_state[n_index + model_integrator.number_of_neurons] =
                model_state[n_index + model_integrator.number_of_neurons] + d;
        }
    }
}

impl IzikevichModel {
    pub fn euler_step(
        &self,
        model_state: &mut IzikevichModelState,
        input_current: &DVector<f64>,
        connectivity_matrix: &DMatrix<f64>,
    ) {
        let firing_neurons = model_state.reset_firing_neurons(self.c, self.d, self.spike_trashhold);
        let adjusted_current = input_current
            + firing_neurons
                .iter()
                .fold(DVector::zeros(self.number_of_neurons), |acc, e| {
                    acc + connectivity_matrix.column(*e)
                });
        model_state.adjust_state(self.dt, adjusted_current, self.a, self.b);
    }
}

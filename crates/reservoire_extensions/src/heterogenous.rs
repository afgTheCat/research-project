use crate::input_step::InputStepsHeterogenous;
use pyo3::prelude::*;
use reservoire::heterogenous_model::HeterogenousReserviore as HeterogenousReservioreRust;

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[pyclass]
pub struct HeterogenousReservoire {
    reservoire: HeterogenousReservioreRust,
}

#[pymethods]
impl HeterogenousReservoire {
    #[new]
    fn new(number_of_neurons: usize, dt: f64) -> Self {
        HeterogenousReservoire {
            reservoire: HeterogenousReservioreRust::new(number_of_neurons, dt),
        }
    }

    fn get_states(&mut self, inputs: InputStepsHeterogenous) -> (Vec<f64>, Vec<Vec<f64>>) {
        let network_states = self.reservoire.integrate_network(inputs.input_steps);
        let times = network_states
            .iter()
            .map(|network_state| network_state.time())
            .collect();

        let membrane_potentials = network_states
            .iter()
            .map(|network_state| network_state.membrane_potentials().to_vec())
            .collect::<Vec<_>>();

        (times, transpose(membrane_potentials))
    }
}

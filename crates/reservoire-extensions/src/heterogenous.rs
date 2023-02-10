use crate::input_step::InputStepsHeterogenous;
use pyo3::prelude::*;
use reservoire::heterogenous_model::HeterogenousReserviore as HeterogenousReservioreRust;

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

    fn get_states(&self, input: InputStepsHeterogenous) {}
}

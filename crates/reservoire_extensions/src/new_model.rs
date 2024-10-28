use core::f64;

use nalgebra::DVector;
use pyo3::{pyclass, pymethods};
use reservoire::new_model::NewModel;

#[pyclass]
pub struct NewModelWrapper {
    reservoire: NewModel,
}

// TODO: PERF
#[pymethods]
impl NewModelWrapper {
    fn diffuse(&mut self, input: Vec<f64>) -> Vec<f64> {
        let excited_input = self.reservoire.diffuse(DVector::from_vec(input));
        excited_input.data.as_vec().clone()
    }

    fn excite(&mut self, input: Vec<f64>, dt: f64) -> Vec<f64> {
        let voltages = self.reservoire.excite(DVector::from_vec(input), dt);
        voltages.data.as_vec().clone()
    }
}


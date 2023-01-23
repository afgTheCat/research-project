use pyo3::prelude::*;
use reservoire::izikevich_model::InputStepHomogenous as InputStepRust;

#[pyclass]
#[derive(Debug, Clone)]
pub struct InputSteps {
    pub input_steps: Vec<InputStepRust>,
}

#[pymethods]
impl InputSteps {
    #[new]
    fn new(input_vals: Vec<(f64, Vec<f64>)>) -> Self {
        let input_steps = input_vals
            .iter()
            .map(|(duration, input_vals)| InputStepRust::new(*duration, input_vals.clone()))
            .collect::<Vec<_>>();
        Self { input_steps }
    }

    fn vals(&self) -> Vec<Vec<f64>> {
        self.input_steps
            .iter()
            .map(|input_step| input_step.vals().into_iter().cloned().collect())
            .collect()
    }
}

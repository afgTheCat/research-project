use pyo3::prelude::*;
use reservoire::heterogenous_model::InputStepHeterogenous as InputStepRustHeterogenous;
use reservoire::izikevich_model::InputStepHomogenous as InputStepRustHomogenous;

#[pyclass]
#[derive(Debug, Clone)]
pub struct InputStepsHomogenous {
    pub input_steps: Vec<InputStepRustHomogenous>,
}

#[pymethods]
impl InputStepsHomogenous {
    #[new]
    fn new(input_vals: Vec<(f64, Vec<f64>)>) -> Self {
        let input_steps = input_vals
            .iter()
            .map(|(duration, input_vals)| {
                InputStepRustHomogenous::new(*duration, input_vals.clone())
            })
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct InputStepsHeterogenous {
    pub input_steps: Vec<InputStepRustHeterogenous>,
}

#[pymethods]
impl InputStepsHeterogenous {
    #[new]
    fn new(input_vals: Vec<(f64, Vec<f64>)>) -> Self {
        let input_steps = input_vals
            .iter()
            .map(|(duration, input_vals)| {
                InputStepRustHeterogenous::new(*duration, input_vals.clone())
            })
            .collect::<Vec<_>>();
        Self { input_steps }
    }

    fn vals(&self) -> Vec<Vec<f64>> {
        self.input_steps
            .iter()
            .map(|input_step| input_step.input().into_iter().cloned().collect())
            .collect()
    }
}

use pyo3::prelude::*;

#[pyfunction]
fn reservoire_test() -> (Vec<f64>, Vec<Vec<f64>>) {
    reservoire::reserviore_test()
}

#[pymodule]
fn reservoire_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reservoire_test, m)?)?;
    Ok(())
}

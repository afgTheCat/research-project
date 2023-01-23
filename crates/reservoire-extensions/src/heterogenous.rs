use pyo3::prelude::*;
use reservoire::heterogenous_model::HeterogenousReserviore as HeterogenousReservioreRust;

#[pyclass]
pub struct HeterogenousReservoire {
    reservoire: HeterogenousReservioreRust,
}

#[pymethods]
impl HeterogenousReservoire {
    #[new]
    fn new() -> Self {
        HeterogenousReservoire {
            reservoire: HeterogenousReservioreRust::new(100, 0.05),
        }
    }
}

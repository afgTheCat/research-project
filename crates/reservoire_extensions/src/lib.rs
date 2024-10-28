mod heterogenous;
mod homogenous;
mod input_step;
mod new_model;

use heterogenous::HeterogenousReservoire;
use homogenous::{
    ConnectivityPrimitive, HomogenousReservoire, InputPrimitive, NetworkInitPrimitive,
    ThalmicPrimitive, VariantChooser,
};
use input_step::{InputStepsHeterogenous, InputStepsHomogenous};
use pyo3::prelude::*;

#[pymodule]
fn reservoire_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<HomogenousReservoire>()?;
    m.add_class::<HeterogenousReservoire>()?;
    m.add_class::<InputStepsHomogenous>()?;
    m.add_class::<InputStepsHeterogenous>()?;
    m.add_class::<VariantChooser>()?;
    m.add_class::<NetworkInitPrimitive>()?;
    m.add_class::<ConnectivityPrimitive>()?;
    m.add_class::<InputPrimitive>()?;
    m.add_class::<ThalmicPrimitive>()?;
    Ok(())
}

mod homogenous;
mod input_step;

use homogenous::{
    ConnectivityPrimitive, InputPrimitive, NetworkInitPrimitive, Reservoire, ThalmicPrimitive,
    VariantChooser,
};
use input_step::InputSteps;
use pyo3::prelude::*;

#[pymodule]
fn reservoire_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Reservoire>()?;
    m.add_class::<InputSteps>()?;
    m.add_class::<VariantChooser>()?;
    m.add_class::<NetworkInitPrimitive>()?;
    m.add_class::<ConnectivityPrimitive>()?;
    m.add_class::<InputPrimitive>()?;
    m.add_class::<ThalmicPrimitive>()?;
    Ok(())
}

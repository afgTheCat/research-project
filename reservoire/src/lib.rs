mod the_reservoire;

use either::Either;
use nalgebra::{Const, Matrix};
use num_complex::Complex;
use pyo3::prelude::*;
pub use the_reservoire::integrate_until_time;
use the_reservoire::{ConnectivityGraphType, NetworkInitialization};


#[pyfunction]
fn test_reservoire() -> (Vec<f64>, Vec<Vec<f64>>) {
    let network_initialization = NetworkInitialization::NoRandomWeight {
        membrane_potential: -65.0,
        recovery_variable: -14.0,
    };
    let connectivity_graph = Either::Left(ConnectivityGraphType::Erdos { connectivity: 1.0 });
    integrate_until_time(1000.0, 10, connectivity_graph, network_initialization)
}

/// A Python module implemented in Rust.
#[pymodule]
fn reservoire(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_reservoire, m)?)?;
    Ok(())
}


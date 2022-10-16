use either::Either;
use pyo3::prelude::*;
use reservoire::izikevich_model::{
    ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, IzikevichModel,
};

#[pyfunction]
fn reservoire_test() -> (Vec<f64>, Vec<Vec<f64>>) {
    reservoire::reserviore_test()
}

#[pyclass]
pub struct Reservoire {
    reservoire: IzikevichModel,
}

#[pymethods]
impl Reservoire {
    #[new]
    fn new_with() -> Self {
        let number_of_neurons = 10;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);
        let connectivity_graph = Either::Left(ConnectivitySetUpType::Erdos { connectivity: 1.0 });
        let input_matrix_setup = InputMatrixSetUp::AllConnected;
        let network_initialization = InitialNetworkStateInit::NoRandomWeight {
            membrane_potential: -65.0,
            recovery_variable: -14.0,
        };
        Self {
            reservoire: IzikevichModel::new(
                a,
                b,
                c,
                d,
                dt,
                number_of_neurons,
                spike_val,
                connectivity_graph,
                network_initialization,
                input_matrix_setup,
            ),
        }
    }

    fn test_run(&self) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        self.reservoire.get_states_with_val()
    }
}

#[pymodule]
fn reservoire_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reservoire_test, m)?)?;
    m.add_class::<Reservoire>()?;
    Ok(())
}

use either::Either;
use nalgebra::DMatrix;
use pyo3::prelude::*;
use reservoire::izikevich_model::InputStep as InputStepRust;
use reservoire::izikevich_model::{
    ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, IzikevichModel,
};

#[pyclass]
pub struct Reservoire {
    reservoire: IzikevichModel,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct InputSteps {
    input_steps: Vec<InputStepRust>,
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
}

#[pyclass]
#[derive(Debug, Clone)]
enum NetworkInitPrimitive {
    NoRandomWeight,
    NormalRandomWeight,
}

impl Default for VariantChooser {
    fn default() -> Self {
        Self {
            network_init_primitive: NetworkInitPrimitive::NoRandomWeight,
            network_membrane_potential: -65.0,
            network_recovery_variable: -14.0,
            network_membrane_potential_dev: 0.0,
            network_recovery_variable_dev: 0.0,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VariantChooser {
    network_init_primitive: NetworkInitPrimitive,
    network_membrane_potential: f64,
    network_recovery_variable: f64,
    network_membrane_potential_dev: f64,
    network_recovery_variable_dev: f64,
}

impl VariantChooser {
    fn connectivity_graph(&self) -> Either<ConnectivitySetUpType, DMatrix<f64>> {
        Either::Left(ConnectivitySetUpType::Erdos(1.0))
    }

    fn input_matrix_setup(&self) -> InputMatrixSetUp {
        InputMatrixSetUp::AllConnected
    }

    fn network_initialization(&self) -> InitialNetworkStateInit {
        match self.network_init_primitive {
            NetworkInitPrimitive::NoRandomWeight => InitialNetworkStateInit::NoRandomWeight {
                membrane_potential: self.network_membrane_potential,
                recovery_variable: self.network_recovery_variable,
            },
            NetworkInitPrimitive::NormalRandomWeight => InitialNetworkStateInit::NormalWeight {
                membrane_potential: self.network_membrane_potential,
                recovery_variable: self.network_recovery_variable,
                membrane_potential_deviation: self.network_membrane_potential_dev,
                recovery_variable_deviation: self.network_recovery_variable_dev,
            },
        }
    }
}

#[pymethods]
impl VariantChooser {
    #[new]
    #[args(
        network_init_primitive = "NetworkInitPrimitive::NoRandomWeight",
        network_membrane_potential = "-65.0",
        network_recovery_variable = "-14.0",
        network_membrane_potential_dev = "0.0",
        network_recovery_variable_dev = "0.0"
    )]
    fn new(
        network_init_primitive: NetworkInitPrimitive,
        network_membrane_potential: f64,
        network_membrane_potential_dev: f64,
        network_recovery_variable: f64,
        network_recovery_variable_dev: f64,
    ) -> Self {
        Self {
            network_init_primitive,
            network_membrane_potential,
            network_recovery_variable,
            network_membrane_potential_dev,
            network_recovery_variable_dev,
        }
    }
}

#[pymethods]
impl Reservoire {
    #[new]
    #[args(
        a = "0.02",
        b = "0.2",
        c = "-65.0",
        d = "8.0",
        dt = "0.05",
        spike_val = "35.0",
        number_of_neurons = "10",
        variant_chooser = "VariantChooser::default()"
    )]
    fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        dt: f64,
        number_of_neurons: usize,
        spike_val: f64,
        variant_chooser: VariantChooser,
    ) -> Self {
        let connectivity_graph = variant_chooser.connectivity_graph();
        let input_matrix_setup = variant_chooser.input_matrix_setup();
        let network_initialization = variant_chooser.network_initialization();
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

    fn get_states(&self, input: InputSteps) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        self.reservoire.get_states(input.input_steps)
    }
}

#[pymodule]
fn reservoire_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Reservoire>()?;
    m.add_class::<InputSteps>()?;
    m.add_class::<NetworkInitPrimitive>()?;
    m.add_class::<VariantChooser>()?;
    Ok(())
}

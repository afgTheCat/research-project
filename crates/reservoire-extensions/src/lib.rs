use log::info;
use nalgebra::DMatrix;
use pyo3::prelude::*;
use reservoire::izikevich_model::InputStep as InputStepRust;
use reservoire::izikevich_model::{
    ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, IzikevichModel, ThalmicInput,
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

    fn vals(&self) -> Vec<Vec<f64>> {
        self.input_steps
            .iter()
            .map(|input_step| input_step.vals().into_iter().cloned().collect())
            .collect()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
enum NetworkInitPrimitive {
    NoRandomWeight,
    NormalRandomWeight,
}

#[pyclass]
#[derive(Debug, Clone)]
enum ConnectivityPrimitive {
    ErdosUniform,
    ErdosNormal,
    ErdosSpectral,
}

#[pyclass]
#[derive(Debug, Clone)]
enum InputPrimitive {
    AllConnected,
    PercentageConnected,
}

#[pyclass]
#[derive(Debug, Clone)]
enum ThalmicPrimitive {
    Const,
    Normal,
}

impl Default for VariantChooser {
    fn default() -> Self {
        Self {
            network_init_primitive: NetworkInitPrimitive::NoRandomWeight,
            connectivity_primitive: ConnectivityPrimitive::ErdosUniform,
            input_primitive: InputPrimitive::AllConnected,
            thalmic_primitive: ThalmicPrimitive::Const,
            input_connectivity_p: 0.5,
            network_membrane_potential: -65.0,
            network_recovery_variable: -14.0,
            network_membrane_potential_dev: 0.0,
            network_recovery_variable_dev: 0.0,
            erdos_connectivity: 1.0,
            erdos_uniform_lower: 0.0,
            erdos_uniform_upper: 1.0,
            erdos_normal_mean: 0.0,
            erdos_normal_dev: 1.0,
            erdos_spectral_radius: 0.59,
            thalmic_mean: 0.0,
            thalmic_dev: 0.0,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VariantChooser {
    network_init_primitive: NetworkInitPrimitive,
    connectivity_primitive: ConnectivityPrimitive,
    input_primitive: InputPrimitive,
    thalmic_primitive: ThalmicPrimitive,
    input_connectivity_p: f64,
    network_membrane_potential: f64,
    network_recovery_variable: f64,
    network_membrane_potential_dev: f64,
    network_recovery_variable_dev: f64,
    erdos_connectivity: f64,
    erdos_uniform_lower: f64,
    erdos_uniform_upper: f64,
    erdos_normal_mean: f64,
    erdos_normal_dev: f64,
    erdos_spectral_radius: f64,
    thalmic_mean: f64,
    thalmic_dev: f64,
}

impl VariantChooser {
    fn connectivity_graph(&self) -> ConnectivitySetUpType {
        match self.connectivity_primitive {
            ConnectivityPrimitive::ErdosUniform => ConnectivitySetUpType::ErdosLowerUpper {
                connectivity: self.erdos_connectivity,
                lower: self.erdos_uniform_lower,
                upper: self.erdos_uniform_upper,
            },
            ConnectivityPrimitive::ErdosNormal => ConnectivitySetUpType::ErdosNormal {
                connectivity: self.erdos_connectivity,
                mean: self.erdos_normal_mean,
                dev: self.erdos_normal_dev,
            },
            ConnectivityPrimitive::ErdosSpectral => ConnectivitySetUpType::ErdosSpectral {
                connectivity: self.erdos_connectivity,
                spectral_radius: self.erdos_spectral_radius,
            },
        }
    }

    fn input_matrix_setup(&self) -> InputMatrixSetUp {
        match self.input_primitive {
            InputPrimitive::AllConnected => InputMatrixSetUp::AllConnected,
            InputPrimitive::PercentageConnected => InputMatrixSetUp::PercentageConnected {
                connectivity: self.input_connectivity_p,
            },
        }
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

    fn thalmic_input(&self) -> ThalmicInput {
        match self.thalmic_primitive {
            ThalmicPrimitive::Const => ThalmicInput::Const(self.thalmic_mean),
            ThalmicPrimitive::Normal => ThalmicInput::Normal {
                mean: self.thalmic_mean,
                dev: self.thalmic_dev,
            },
        }
    }
}

#[pymethods]
impl VariantChooser {
    #[new]
    #[args(
        network_init_primitive = "NetworkInitPrimitive::NoRandomWeight",
        connectivity_primitive = "ConnectivityPrimitive::ErdosUniform",
        input_primitive = "InputPrimitive::AllConnected",
        network_membrane_potential = "-65.0",
        network_recovery_variable = "-14.0",
        network_membrane_potential_dev = "0.0",
        network_recovery_variable_dev = "0.0",
        erdos_connectivity = "1.0",
        erdos_uniform_lower = "0.0",
        erdos_uniform_upper = "1.0",
        erdos_normal_mean = "0.0",
        erdos_normal_dev = "1.0",
        erdos_spectral_radius = "0.59",
        input_connectivity_p = "0.5",
        thalmic_mean = "0.0",
        thalmic_dev = "0.0"
    )]
    fn new(
        network_init_primitive: NetworkInitPrimitive,
        connectivity_primitive: ConnectivityPrimitive,
        thalmic_primitive: ThalmicPrimitive,
        input_primitive: InputPrimitive,
        network_membrane_potential: f64,
        network_membrane_potential_dev: f64,
        network_recovery_variable: f64,
        network_recovery_variable_dev: f64,
        erdos_connectivity: f64,
        erdos_uniform_lower: f64,
        erdos_uniform_upper: f64,
        erdos_normal_mean: f64,
        erdos_normal_dev: f64,
        erdos_spectral_radius: f64,
        input_connectivity_p: f64,
        thalmic_mean: f64,
        thalmic_dev: f64,
    ) -> Self {
        Self {
            network_init_primitive,
            connectivity_primitive,
            thalmic_primitive,
            input_primitive,
            network_membrane_potential,
            network_recovery_variable,
            network_membrane_potential_dev,
            network_recovery_variable_dev,
            erdos_connectivity,
            erdos_uniform_lower,
            erdos_uniform_upper,
            erdos_normal_mean,
            erdos_normal_dev,
            erdos_spectral_radius,
            input_connectivity_p,
            thalmic_mean,
            thalmic_dev,
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
        info!("connectivity setup type: {:#x?}", connectivity_graph);
        let input_matrix_setup = variant_chooser.input_matrix_setup();
        info!("input matrix setup: {:x?}", input_matrix_setup);
        let network_initialization = variant_chooser.network_initialization();
        info!("initial network init: {:x?}", network_initialization);
        let thalmic_input = variant_chooser.thalmic_input();
        info!("thalmic input init: {:x?}", thalmic_input);
        let reservoire = IzikevichModel::new(
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
            thalmic_input,
        );
        Self { reservoire }
    }

    fn get_states(&self, input: InputSteps) -> (Vec<f64>, Vec<Vec<f64>>) {
        self.reservoire.get_states(input.input_steps)
    }
}

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

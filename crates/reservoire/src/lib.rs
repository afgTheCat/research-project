pub mod heterogenous_model;
mod integration;
pub mod izikevich_model;

#[cfg(test)]
mod test {
    use crate::{
        heterogenous_model::{Network, Parameters},
        izikevich_model::{
            ConnectivitySetUpType, InputMatrixSetUp, InputStep, IzhikevichModel, NetworkInit,
            ThalmicInput,
        },
    };

    #[test]
    fn reserviore_test() {
        let _ = env_logger::try_init();

        let number_of_neurons = 100;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);

        let connectivity_graph = ConnectivitySetUpType::ErdosZeroOne(1.0);
        let input_matrix_setup = InputMatrixSetUp::PercentageConnected { connectivity: 0.3 };

        let network_initialization = NetworkInit::NormalWeight {
            membrane_potential: -65.0,
            recovery_variable: -14.0,
            membrane_potential_deviation: 10.0,
            recovery_variable_deviation: 3.0,
        };
        let thalmic_input = ThalmicInput::Const(10.0);
        let izikevich_model = IzhikevichModel::new(
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
        let inputs = vec![InputStep::new(10.0, vec![0.5; 20]); 2];
        let states = izikevich_model.get_states(inputs);
        log::info!("states: {:#x?}", states);
    }

    #[test]
    fn heterogenous_mode() {
        let _ = env_logger::try_init();
        let (a, b, c, d) = (0.02, 0.2, -65.0, 8.0);
        let parameters = vec![Parameters::new(a, b, c, d)];
        let connectivity_setup = ConnectivitySetUpType::ErdosLowerUpper {
            connectivity: 0.3,
            lower: 0.0,
            upper: 2.0,
        };
        let network = Network::new(parameters, connectivity_setup);
    }
}

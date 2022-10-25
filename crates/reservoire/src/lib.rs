mod integration;
pub mod izikevich_model;

#[cfg(test)]
mod test {
    use crate::izikevich_model::{
        ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, InputStep,
        IzikevichModel, ThalmicInput,
    };
    use either::Either;

    #[test]
    fn reserviore_test() {
        let _ = env_logger::try_init();

        let number_of_neurons = 100;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);

        let connectivity_graph = ConnectivitySetUpType::ErdosZeroOne(1.0);
        let input_matrix_setup = InputMatrixSetUp::PercentageConnected { connectivity: 0.3 };

        let network_initialization = InitialNetworkStateInit::NormalWeight {
            membrane_potential: -65.0,
            recovery_variable: -14.0,
            membrane_potential_deviation: 10.0,
            recovery_variable_deviation: 3.0,
        };

        let thalmic_input = ThalmicInput::Const(10.0);

        let izikevich_model = IzikevichModel::new(
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
}

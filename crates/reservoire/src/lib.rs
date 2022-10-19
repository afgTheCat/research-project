mod integration;
pub mod izikevich_model;

#[cfg(test)]
mod test {
    use crate::izikevich_model::{
        ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, InputStep, IzikevichModel,
    };
    use either::Either;

    #[test]
    fn reserviore_test_one() {
        let _ = env_logger::try_init();

        let number_of_neurons = 1;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);
        let connectivity_graph = Either::Left(ConnectivitySetUpType::ErdosZeroOne(0.25));
        let input_matrix_setup = InputMatrixSetUp::AllConnected;

        let network_initialization = InitialNetworkStateInit::NoRandomWeight {
            membrane_potential: -65.0,
            recovery_variable: -14.0,
        };

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
        );
        let inputs = vec![InputStep::new(1000.0, vec![10.0]); 1];
        if let Some(states) = izikevich_model.get_states_chemical(inputs) {
            // log::info!("this thing: {:#x?}", states);
        } else {
            log::info!("something went wrong!!");
        }
    }

    #[test]
    fn reserviore_test_two() {
        let _ = env_logger::try_init();

        let number_of_neurons = 10;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);
        let connectivity_graph = Either::Left(ConnectivitySetUpType::ErdosZeroOne(0.25));
        let input_matrix_setup = InputMatrixSetUp::AllConnected;

        let network_initialization = InitialNetworkStateInit::NormalWeight {
            membrane_potential: -65.0,
            recovery_variable: -14.0,
            membrane_potential_deviation: 10.0,
            recovery_variable_deviation: 3.0,
        };

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
        );
        let inputs = vec![InputStep::new(1000.0, vec![10.0]); 1];
        if let Some(states) = izikevich_model.get_states(inputs) {
            // log::info!("this thing: {:#x?}", states);
        } else {
            log::info!("something went wrong!!");
        }
    }
}

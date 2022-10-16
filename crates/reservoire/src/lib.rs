mod integration;
pub mod izikevich_model;

use crate::izikevich_model::{
    ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, InputStep, IzikevichModel,
};
use either::Either;
use nalgebra::DVector;

pub fn reserviore_test() -> (Vec<f64>, Vec<Vec<f64>>) {
    let number_of_neurons = 10;
    let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);
    let connectivity_graph = Either::Left(ConnectivitySetUpType::Erdos { connectivity: 1.0 });
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
    let input_vals = DVector::from_vec(vec![10.0]);
    let inputs = vec![InputStep::new(1000.0, input_vals)];
    // TODO: should this ever fail?
    izikevich_model.get_states(inputs).unwrap()
}

#[cfg(test)]
mod test {
    use crate::izikevich_model::{
        ConnectivitySetUpType, InitialNetworkStateInit, InputMatrixSetUp, InputStep, IzikevichModel,
    };
    use either::Either;
    use nalgebra::DVector;

    #[test]
    fn reserviore_test_one() {
        let _ = env_logger::try_init();

        let number_of_neurons = 10;
        let (a, b, c, d, dt, spike_val) = (0.02, 0.2, -65.0, 8.0, 0.05, 35.0);
        let connectivity_graph = Either::Left(ConnectivitySetUpType::Erdos { connectivity: 1.0 });
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
        let input_vals = DVector::from_vec(vec![10.0]);
        let inputs = vec![InputStep::new(1000.0, input_vals)];
        if let Some(states) = izikevich_model.get_states(inputs) {
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
        let connectivity_graph = Either::Left(ConnectivitySetUpType::Erdos { connectivity: 1.0 });
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
        let input_vals = DVector::from_vec(vec![1.0; 10]);
        let inputs = vec![InputStep::new(1000.0, input_vals)];
        if let Some(states) = izikevich_model.get_states(inputs) {
            // log::info!("this thing: {:#x?}", states);
        } else {
            log::info!("something went wrong!!");
        }
    }
}

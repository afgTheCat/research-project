use either::Either;
use nalgebra::SMatrix;
use ode_solvers::System;
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::Normal;
use std::{iter::repeat_with, usize};

#[derive(Debug, Clone, Copy)]
pub struct IzikevichNeuron {
    membrane_potential: f64,
    recovery_variable: f64,
}

impl IzikevichNeuron {
    fn new(membrane_potential: f64, recovery_variable: f64) -> Self {
        Self {
            membrane_potential,
            recovery_variable,
        }
    }
}

pub struct IzikevichNetwork<const N: usize> {
    izikevich_neurons: [IzikevichNeuron; N],
    connectivity_matrix: SMatrix<f64, N, N>,
}

enum NetworkInitialization {
    NoRandomWeight {
        membrane_potential: f64,
        recovery_variable: f64,
    },
    NormalWeight {
        membrane_potential: f64,
        recovery_variable: f64,
        membrane_potential_deviation: f64,
        recovery_variable_deviation: f64,
    },
}

// TODO: more strategies!
enum ConnectivityGraphType {
    Erdos { connectivity: f64 },
}

fn create_random_graph<const N: usize>(
    connectivity_graph_type: ConnectivityGraphType,
) -> SMatrix<f64, N, N> {
    match connectivity_graph_type {
        ConnectivityGraphType::Erdos { connectivity } => {
            let mut rng = rand::thread_rng();
            let bernoulli_distr = Bernoulli::new(connectivity).unwrap();
            let erdos_iter = repeat_with(|| {
                if bernoulli_distr.sample(&mut rng) {
                    rng.gen_range(0.0..1.0)
                } else {
                    0.0
                }
            });
            SMatrix::from_iterator(erdos_iter)
        }
    }
}

fn create_initial_neurons<const N: usize>(
    init_network: NetworkInitialization,
) -> [IzikevichNeuron; N] {
    match init_network {
        NetworkInitialization::NoRandomWeight {
            membrane_potential,
            recovery_variable,
        } => {
            // this can be slow
            [IzikevichNeuron::new(membrane_potential, recovery_variable); N]
        }
        NetworkInitialization::NormalWeight {
            membrane_potential,
            recovery_variable,
            membrane_potential_deviation,
            recovery_variable_deviation,
        } => {
            let mut rng = rand::thread_rng();
            let membrane_potential_distribution =
                Normal::new(membrane_potential, membrane_potential_deviation).unwrap();
            let recovery_variable_distribution =
                Normal::new(recovery_variable, recovery_variable_deviation).unwrap();
            let neuron_iter = repeat_with(|| {
                let membrane_potential = membrane_potential_distribution.sample(&mut rng);
                let recovery_variable = recovery_variable_distribution.sample(&mut rng);
                IzikevichNeuron::new(membrane_potential, recovery_variable)
            })
            .take(N);
            neuron_iter.collect::<Vec<_>>().try_into().unwrap()
        }
    }
}

impl<const N: usize> IzikevichNetwork<N> {
    fn new(
        network_initialization: Either<[IzikevichNeuron; N], NetworkInitialization>,
        connectivity_initialization: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
    ) -> Self {
        let izikevich_neurons = match network_initialization {
            Either::Left(neurons) => neurons,
            Either::Right(init_network) => create_initial_neurons(init_network),
        };
        let connectivity_matrix = match connectivity_initialization {
            Either::Left(connectivity_graph_type) => create_random_graph(connectivity_graph_type),
            Either::Right(connectivity_matrix) => connectivity_matrix,
        };
        Self {
            izikevich_neurons,
            connectivity_matrix,
        }
    }
}

pub struct IzikevichModel<const N: usize> {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    network: IzikevichNetwork<N>,
}

impl<const N: usize> IzikevichModel<N> {
    fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        network_initialization: Either<[IzikevichNeuron; N], NetworkInitialization>,
        connectivity_initialization: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
    ) -> Self {
        let network = IzikevichNetwork::new(network_initialization, connectivity_initialization);
        Self {
            a,
            b,
            c,
            d,
            network,
        }
    }
}

type IzikevichModelState<const N: usize> = SMatrix<f64, N, 2>;

impl<const N: usize> System<IzikevichModelState<N>> for IzikevichModel<N> {
    fn system(&self, _time: f64, y: &IzikevichModelState<N>, dy: &mut IzikevichModelState<N>) {
        // TODO:
    }

    fn solout(
        &mut self,
        _x: f64,
        _y: &IzikevichModelState<N>,
        _dy: &IzikevichModelState<N>,
    ) -> bool {
        // TODO: Stop and restart on
        false
    }
}

#[cfg(test)]
mod test {}

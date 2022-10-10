#![feature(generic_const_exprs)]

use either::Either;
use nalgebra::{SMatrix, Vector};
use ode_solvers::{Dopri5, System};
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::{num_traits::Pow, Normal};
use std::{
    fs::File,
    io::{BufWriter, Write},
    iter::repeat_with,
    path::Path,
    usize,
};

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

#[derive(Debug, Clone)]
pub struct IzikevichNetwork<const N: usize> {
    izikevich_neurons: [IzikevichNeuron; N],
    connectivity_matrix: SMatrix<f64, N, N>,
}

pub enum NetworkInitialization {
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

pub enum ConnectivityGraphType {
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
        connectivity_graph: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
    ) -> Self {
        let izikevich_neurons = match network_initialization {
            Either::Left(neurons) => neurons,
            Either::Right(init_network) => create_initial_neurons(init_network),
        };
        let connectivity_matrix = match connectivity_graph {
            Either::Left(connectivity_graph_type) => create_random_graph(connectivity_graph_type),
            Either::Right(connectivity_matrix) => connectivity_matrix,
        };
        Self {
            izikevich_neurons,
            connectivity_matrix,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IzikevichModel<const N: usize> {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    spike_value: f64,
    // network: IzikevichNetwork<N>,
    connectivity_matrix: SMatrix<f64, N, N>,
}

impl<const N: usize> IzikevichModel<N> {
    fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        spike_value: f64,
        // network_initialization: Either<[IzikevichNeuron; N], NetworkInitialization>,
        // connectivity_initialization: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
        connectivity_graph: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
    ) -> Self {
        // let network = IzikevichNetwork::new(network_initialization, connectivity_initialization);

        let connectivity_matrix = match connectivity_graph {
            Either::Left(connectivity_graph_type) => create_random_graph(connectivity_graph_type),
            Either::Right(connectivity_matrix) => connectivity_matrix,
        };
        Self {
            a,
            b,
            c,
            d,
            spike_value,
            connectivity_matrix, // network,
        }
    }

    // TODO: this should be a function of time
    fn current(&self, time: f64) -> f64 {
        10.0
    }

    pub fn initial_state(&self) -> IzikevichModelState<{ 2 * N }> {
        let initial_state = [-70.0; N]
            .iter()
            .chain([-14.0; N].iter())
            .cloned()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        IzikevichModelState::from_vec(initial_state)
    }

    pub fn handle_post_fire(&self, post_fire: &mut IzikevichModelState<{ 2 * N }>) {
        for n_index in 0..N {
            if post_fire[n_index] > self.spike_value {
                post_fire[n_index] = self.c;
                post_fire[n_index + N] = post_fire[n_index + N] + self.d;
            }
        }
    }
}

type IzikevichModelState<const N: usize> = SMatrix<f64, N, 1>;

impl<const N: usize> System<IzikevichModelState<{ 2 * N }>> for IzikevichModel<N> {
    fn system(
        &self,
        time: f64,
        y: &IzikevichModelState<{ 2 * N }>,
        dy: &mut IzikevichModelState<{ 2 * N }>,
    ) {
        let current = SMatrix::<f64, N, 1>::zeros().add_scalar(self.current(time));
        let v_slice = y.fixed_slice::<N, 1>(0, 0);
        let u_slice = y.fixed_slice::<N, 1>(N, 0);
        let new_v_slice = ((0.04 * v_slice.component_mul(&v_slice)) + (v_slice * 5.0))
            .add_scalar(140.0)
            - u_slice
            + current;
        let new_u_slice = self.a * (self.b * v_slice - u_slice);

        let mut dv_slice = dy.fixed_slice_mut::<N, 1>(0, 0);
        dv_slice.set_column(0, &new_v_slice);
        let mut du_slice = dy.fixed_slice_mut::<N, 1>(N, 0);
        du_slice.set_column(0, &new_u_slice);

        // TODO: maybe we can do this better?
        for i in 0..N {
            // self.connectivity_graph.row(i)
        }
    }

    fn solout(
        &mut self,
        _time: f64,
        y: &IzikevichModelState<{ 2 * N }>,
        _dy: &IzikevichModelState<{ 2 * N }>,
    ) -> bool {
        let ehh = y.iter().any(|x| x > &self.spike_value);
        ehh
    }
}

pub fn save<const N: usize>(
    times: &Vec<f64>,
    states: &Vec<IzikevichModelState<N>>,
    filename: &Path,
) {
    let file = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };
    let mut buf = BufWriter::new(file);
    for (i, state) in states.iter().enumerate() {
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        for val in state.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        buf.write_fmt(format_args!("\n")).unwrap();
    }
    if let Err(e) = buf.flush() {
        println!("Could not write to file. Error: {:?}", e);
    }
}

pub fn integrate_until_time<const N: usize>(
    t_last: f64,
    // network_initialization: Either<[IzikevichNeuron; N], NetworkInitialization>,
    connectivity_graph: Either<ConnectivityGraphType, SMatrix<f64, N, N>>,
) where
    [f64; 2 * N]:,
{
    let izikevich_model = IzikevichModel::new(
        0.02,
        0.2,
        -65.0,
        8.0,
        35.0,
        // network_initialization,
        connectivity_graph,
    );
    let mut t = 0.0;
    let dt = 0.005;
    let mut times: Vec<f64> = vec![];
    let mut states: Vec<IzikevichModelState<{ 2 * N }>> = vec![];

    let mut state = izikevich_model.initial_state();
    while t < t_last - dt {
        let mut stepper = Dopri5::new(
            izikevich_model.clone(),
            t,
            t_last,
            dt,
            state,
            1.0e-10,
            1.0e-10,
        );
        let res = stepper.integrate();

        match res {
            Ok(stats) => {
                let times_len = stepper.x_out().len();

                times.extend(stepper.x_out().iter().take(times_len - 1).cloned());
                states.extend(stepper.y_out().iter().take(times_len - 1).cloned());
                t = *stepper.x_out().last().unwrap();

                let mut last_stage = stepper.y_out().last().unwrap().clone();
                izikevich_model.handle_post_fire(&mut last_stage);
                state = last_stage as IzikevichModelState<{ 2 * N }>;
            }
            Err(..) => {
                break;
            }
        }
    }

    let path = Path::new("./outputs/izikevich_model.dat");
    save(&times, &states, path);
}

#[cfg(test)]
mod test {
    use crate::{
        integrate_until_time, ConnectivityGraphType, IzikevichNeuron, NetworkInitialization,
    };
    use either::Either;

    #[test]
    fn test_single_neuron() {
        let _ = env_logger::try_init();
        const N: usize = 1;
        let network_initialization: Either<[IzikevichNeuron; N], NetworkInitialization> =
            Either::Right(NetworkInitialization::NoRandomWeight {
                membrane_potential: -65.0,
                recovery_variable: -14.0,
            });
        let connectivity_graph = Either::Left(ConnectivityGraphType::Erdos { connectivity: 1.0 });
        integrate_until_time::<N>(
            1000.0,
            // network_initialization,
            connectivity_graph,
        );
    }

    #[test]
    fn test_double_neuron() {}
}

use either::Either;
use nalgebra::{DMatrix, DVector};
use ode_solvers::{Dopri5, System};
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::Normal;
use std::iter::repeat_with;

#[derive(Debug, Clone)]
pub enum InitialNetworkStateInit {
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

#[derive(Debug, Clone)]
pub enum InputMatrixSetUp {
    AllConnected,
}

pub enum ConnectivityGraphType {
    Erdos { connectivity: f64 },
}

fn create_random_graph(
    connectivity_graph_type: ConnectivityGraphType,
    size: usize,
) -> DMatrix<f64> {
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
            })
            .take(size * size);
            DMatrix::from_iterator(size, size, erdos_iter)
        }
    }
}

pub struct InputStep {
    duration: f64,
    vals: DMatrix<f64>,
}

#[derive(Debug, Clone)]
pub struct IzikevichModel {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    number_of_neurons: usize,
    spike_value: f64,
    connectivity_matrix: DMatrix<f64>,
    network_initialization: InitialNetworkStateInit,
    input_matrix_setup: InputMatrixSetUp,
}

impl IzikevichModel {
    fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        number_of_neurons: usize,
        spike_value: f64,
        connectivity_graph: Either<ConnectivityGraphType, DMatrix<f64>>,
        network_initialization: InitialNetworkStateInit,
        input_matrix_setup: InputMatrixSetUp,
    ) -> Self {
        let connectivity_matrix = match connectivity_graph {
            Either::Left(connectivity_graph_type) => {
                create_random_graph(connectivity_graph_type, number_of_neurons)
            }
            Either::Right(connectivity_matrix) => connectivity_matrix,
        };
        Self {
            a,
            b,
            c,
            d,
            number_of_neurons,
            spike_value,
            connectivity_matrix,
            network_initialization,
            input_matrix_setup,
        }
    }

    // TODO: this should be a function of time
    fn current(&self, _time: f64) -> f64 {
        10.0
    }

    fn integrate_single_state(
        &self,
        input: InputStep,
        mut current_time: f64,
        dt: f64,
        mut current_state: IzikevichModelState,
        times: &mut Vec<f64>,
        states: &mut Vec<IzikevichModelState>,
    ) -> Option<IzikevichModelState> {
        let time_to_stop = current_time + input.duration;
        loop {
            let mut stepper = Dopri5::new(
                self.clone(),
                current_time,
                time_to_stop,
                dt,
                current_state,
                1.0e-10,
                1.0e-10,
            );
            let res = stepper.integrate();
            match res {
                Ok(..) => {
                    let stepped_lens = stepper.x_out().len();
                    times.extend(stepper.x_out().iter().take(stepped_lens - 1).cloned());
                    states.extend(stepper.y_out().iter().take(stepped_lens - 1).cloned());
                    current_time = *stepper.x_out().last().unwrap();
                    let mut last_stage = stepper.y_out().last().unwrap().clone();
                    self.handle_post_fire(&mut last_stage);
                    current_state = last_stage as IzikevichModelState;
                }
                Err(..) => {
                    break None;
                }
            }
            if current_time > time_to_stop - dt {
                break Some(current_state);
            }
        }
    }

    pub fn get_states(&self, inputs: Vec<InputStep>) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let current_time = 0.0;
        let dt = 0.05;
        let mut times: Vec<f64> = vec![];
        let mut states: Vec<IzikevichModelState> = vec![];
        let mut current_state = self.init_state();
        for input in inputs {
            current_state = self.integrate_single_state(
                input,
                current_time,
                dt,
                current_state,
                &mut times,
                &mut states,
            )?;
        }
        let mut neuron_voltages = repeat_with(|| vec![])
            .take(self.number_of_neurons)
            .collect::<Vec<Vec<f64>>>();

        states.iter().for_each(|time_step| {
            for neuron in 0..self.number_of_neurons {
                neuron_voltages[neuron].push(time_step[neuron]);
            }
        });
        Some((times, neuron_voltages))
    }

    pub fn init_state(&self) -> IzikevichModelState {
        match self.network_initialization {
            InitialNetworkStateInit::NoRandomWeight {
                membrane_potential,
                recovery_variable,
            } => {
                let neuron_iter = repeat_with(|| membrane_potential)
                    .take(self.number_of_neurons)
                    .chain(repeat_with(|| recovery_variable).take(self.number_of_neurons));
                IzikevichModelState::from_iterator(2 * self.number_of_neurons, neuron_iter)
            }
            InitialNetworkStateInit::NormalWeight {
                membrane_potential,
                recovery_variable,
                membrane_potential_deviation,
                recovery_variable_deviation,
            } => {
                let mut membrane_rng = rand::thread_rng();
                let mut recovery_rng = rand::thread_rng();
                let membrane_potential_distribution =
                    Normal::new(membrane_potential, membrane_potential_deviation).unwrap();
                let recovery_variable_distribution =
                    Normal::new(recovery_variable, recovery_variable_deviation).unwrap();
                let neuron_iter =
                    repeat_with(|| membrane_potential_distribution.sample(&mut membrane_rng))
                        .take(self.number_of_neurons)
                        .chain(
                            repeat_with(|| {
                                recovery_variable_distribution.sample(&mut recovery_rng)
                            })
                            .take(self.number_of_neurons),
                        );
                IzikevichModelState::from_iterator(2 * self.number_of_neurons, neuron_iter)
            }
        }
    }

    pub fn handle_post_fire(&self, post_fire: &mut IzikevichModelState) {
        for n_index in 0..self.number_of_neurons {
            if post_fire[n_index] > self.spike_value {
                post_fire[n_index] = self.c;
                post_fire[n_index + self.number_of_neurons] =
                    post_fire[n_index + self.number_of_neurons] + self.d;
            }
        }
    }
}

type IzikevichModelState = DVector<f64>;

impl System<IzikevichModelState> for IzikevichModel {
    fn system(&self, time: f64, y: &IzikevichModelState, dy: &mut IzikevichModelState) {
        // this current has to be changed
        let current = DVector::<f64>::zeros(self.number_of_neurons).add_scalar(self.current(time));

        let v_slice = y.slice((0, 0), (self.number_of_neurons, 1));
        let u_slice = y.slice((self.number_of_neurons, 0), (self.number_of_neurons, 1));
        let new_v_slice: DVector<f64> =
            ((0.04 * v_slice.component_mul(&v_slice)) + (v_slice * 5.0)).add_scalar(140.0)
                - u_slice
                + current;
        let new_u_slice = self.a * (self.b * v_slice - u_slice);
        let mut dv_slice = dy.slice_mut((0, 0), (self.number_of_neurons, 1));
        dv_slice.set_column(0, &new_v_slice);
        let mut du_slice = dy.slice_mut((self.number_of_neurons, 0), (self.number_of_neurons, 1));
        du_slice.set_column(0, &new_u_slice.column(0));

        for i in 0..self.number_of_neurons {
            let v_i = y[i];
            let w = self
                .connectivity_matrix
                .slice((i, 0), (1, self.number_of_neurons))
                * v_slice.add_scalar(-v_i);
            dy[i] += w[(0, 0)];
        }
    }

    fn solout(&mut self, _time: f64, y: &IzikevichModelState, _dy: &IzikevichModelState) -> bool {
        y.iter().any(|x| x > &self.spike_value)
    }
}

struct Integrator {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    number_of_neurons: usize,
    spike_value: f64,
    current_input: DVector<f64>,
}

pub fn integrate_until_time(
    t_last: f64,
    number_of_neurons: usize,
    connectivity_graph: Either<ConnectivityGraphType, DMatrix<f64>>,
    network_initialization: InitialNetworkStateInit,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let izikevich_model = IzikevichModel::new(
        0.02,
        0.2,
        -65.0,
        8.0,
        number_of_neurons,
        35.0,
        connectivity_graph,
        network_initialization,
        InputMatrixSetUp::AllConnected,
    );
    let mut t = 0.0;
    let dt = 0.05;
    let mut times: Vec<f64> = vec![];
    let mut states: Vec<IzikevichModelState> = vec![];
    let mut current_state = izikevich_model.init_state();

    while t < t_last - dt {
        let mut stepper = Dopri5::new(
            izikevich_model.clone(),
            t,
            t_last,
            dt,
            current_state,
            1.0e-10,
            1.0e-10,
        );
        let res = stepper.integrate();

        match res {
            Ok(..) => {
                let times_len = stepper.x_out().len();

                times.extend(stepper.x_out().iter().take(times_len - 1).cloned());
                states.extend(stepper.y_out().iter().take(times_len - 1).cloned());
                t = *stepper.x_out().last().unwrap();

                let mut last_stage = stepper.y_out().last().unwrap().clone();
                izikevich_model.handle_post_fire(&mut last_stage);
                current_state = last_stage as IzikevichModelState;
            }
            Err(..) => {
                break;
            }
        }
    }

    let mut neuron_voltages = repeat_with(|| vec![])
        .take(number_of_neurons)
        .collect::<Vec<Vec<f64>>>();

    states.iter().for_each(|time_step| {
        for neuron in 0..number_of_neurons {
            neuron_voltages[neuron].push(time_step[neuron]);
        }
    });

    (times, neuron_voltages)
}

pub fn reserviore_test() -> (Vec<f64>, Vec<Vec<f64>>) {
    let network_initialization = InitialNetworkStateInit::NoRandomWeight {
        membrane_potential: -65.0,
        recovery_variable: -14.0,
    };
    let connectivity_graph = Either::Left(ConnectivityGraphType::Erdos { connectivity: 1.0 });
    integrate_until_time(1000.0, 10, connectivity_graph, network_initialization)
}

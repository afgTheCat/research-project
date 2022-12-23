use nalgebra::{ComplexField, DMatrix, DVector};
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::Normal;
use std::{
    collections::{HashMap, HashSet},
    iter::repeat_with,
};

use crate::heterogenous_model::ConnectionTarget;

#[derive(Debug, Clone)]
pub struct IzhikevichModelState {
    membrane_potentials: DVector<f64>,
    membrane_recovery_variables: DVector<f64>,
}

impl IzhikevichModelState {
    pub fn reset_firing_neurons(&mut self, c: f64, d: f64, trashhold_value: f64) -> Vec<usize> {
        self.membrane_potentials
            .iter_mut()
            .enumerate()
            .filter_map(|(index, neuron)| {
                if *neuron > trashhold_value {
                    self.membrane_recovery_variables[index] += d;
                    *neuron = c;
                    Some(index)
                } else {
                    None
                }
            })
            .collect()
    }

    fn dmembrane_potentials(&self, input: &DVector<f64>) -> DVector<f64> {
        let membrane_potentials = &self.membrane_potentials;
        let membrane_recovery_variables = &self.membrane_recovery_variables;
        let mut dmembrane_potentials =
            0.04 * membrane_potentials.component_mul(membrane_potentials);
        dmembrane_potentials += 5.0 * membrane_potentials;
        dmembrane_potentials -= membrane_recovery_variables;
        dmembrane_potentials += input;
        dmembrane_potentials.add_scalar(140.0)
    }

    pub fn adjust_state(&mut self, dt: f64, input: DVector<f64>, a: f64, b: f64) {
        self.membrane_potentials += (dt / 2.0) * self.dmembrane_potentials(&input);
        self.membrane_potentials += (dt / 2.0) * self.dmembrane_potentials(&input);
        self.membrane_recovery_variables += dt
            * a
            * (b * self.membrane_potentials.clone() - self.membrane_recovery_variables.clone());
    }
}

#[derive(Debug, Clone)]
pub enum NetworkInit {
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
    PercentageConnected { connectivity: f64 },
}

#[derive(Debug, Clone)]
pub enum ConnectivitySetUpType {
    ErdosZeroOne(f64),
    ErdosZeroUp {
        connectivity: f64,
        upper: f64,
    },
    ErdosLowerUpper {
        connectivity: f64,
        lower: f64,
        upper: f64,
    },
    ErdosNormal {
        connectivity: f64,
        mean: f64,
        dev: f64,
    },
    ErdosSpectral {
        connectivity: f64,
        spectral_radius: f64,
    },
}

impl ConnectivitySetUpType {
    pub fn connections(&self, num_neurons: usize) -> HashMap<usize, Vec<ConnectionTarget>> {
        match &self {
            Self::ErdosLowerUpper {
                connectivity,
                lower,
                upper,
            } => {
                let mut bernoulli_rng = rand::thread_rng();
                let mut uniform_rng = rand::thread_rng();
                let bernoulli_distr = Bernoulli::new(*connectivity).unwrap();
                let mut erdos_iter = repeat_with(|| {
                    if bernoulli_distr.sample(&mut bernoulli_rng) {
                        uniform_rng.gen_range(*lower..*upper)
                    } else {
                        0.0
                    }
                });
                (0..num_neurons)
                    .map(|index| {
                        let mut targets = Vec::new();
                        (0..num_neurons).for_each(|target| {
                            if target != index {
                                let connection_weight = erdos_iter.next().unwrap().clone();
                                if connection_weight != 0.0 {
                                    targets.push(ConnectionTarget::new(connection_weight, target));
                                }
                            }
                        });
                        (index, targets)
                    })
                    .collect::<HashMap<usize, Vec<ConnectionTarget>>>()
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ThalmicInput {
    Const(f64),
    Normal { mean: f64, dev: f64 },
}

fn create_erdos_uniform_connectivity_matrix(
    size: usize,
    connectivity: f64,
    lower: f64,
    upper: f64,
) -> DMatrix<f64> {
    let mut bernoulli_rng = rand::thread_rng();
    let mut uniform_rng = rand::thread_rng();
    let bernoulli_distr = Bernoulli::new(connectivity).unwrap();
    let erdos_iter = repeat_with(|| {
        if bernoulli_distr.sample(&mut bernoulli_rng) {
            uniform_rng.gen_range(lower..upper)
        } else {
            0.0
        }
    })
    .take(size * size);
    DMatrix::from_iterator(size, size, erdos_iter)
}

fn create_erdos_normal_connectivity_matrix(
    size: usize,
    connectivity: f64,
    mean: f64,
    dev: f64,
) -> DMatrix<f64> {
    let mut bernoulli = rand::thread_rng();
    let mut normal_rng = rand::thread_rng();
    let normal_distribution = Normal::new(mean, dev).unwrap();
    let bernoulli_distr = Bernoulli::new(connectivity).unwrap();
    let erdos_iter = repeat_with(|| {
        if bernoulli_distr.sample(&mut bernoulli) {
            normal_distribution.sample(&mut normal_rng)
        } else {
            0.0
        }
    })
    .take(size * size);
    DMatrix::from_iterator(size, size, erdos_iter)
}

#[derive(Debug, Clone)]
pub struct InputStep {
    duration: f64,
    vals: DVector<f64>, // now this is going to be the hardest one yet!
}

impl InputStep {
    pub fn vals(&self) -> &DVector<f64> {
        &self.vals
    }

    fn input_size(&self) -> usize {
        self.vals.len()
    }

    pub fn new(duration: f64, vals: Vec<f64>) -> Self {
        Self {
            duration,
            vals: DVector::from_vec(vals),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IzhikevichModel {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub number_of_neurons: usize,
    pub spike_trashhold: f64,
    pub dt: f64,
    pub connectivity_matrix: DMatrix<f64>,
    pub input_vector: Vec<f64>,
    pub thalmic_input: ThalmicInput,
    network_initialization: NetworkInit,
}

pub fn connectivity_matrix(
    number_of_neurons: usize,
    connectivity_setup: ConnectivitySetUpType,
) -> DMatrix<f64> {
    match connectivity_setup {
        ConnectivitySetUpType::ErdosZeroOne(connectivity) => {
            create_erdos_uniform_connectivity_matrix(number_of_neurons, connectivity, 0.0, 1.0)
        }
        ConnectivitySetUpType::ErdosZeroUp {
            connectivity,
            upper,
        } => create_erdos_uniform_connectivity_matrix(number_of_neurons, connectivity, 0.0, upper),
        ConnectivitySetUpType::ErdosLowerUpper {
            connectivity,
            lower,
            upper,
        } => {
            create_erdos_uniform_connectivity_matrix(number_of_neurons, connectivity, lower, upper)
        }
        ConnectivitySetUpType::ErdosNormal {
            connectivity,
            mean,
            dev,
        } => create_erdos_normal_connectivity_matrix(number_of_neurons, connectivity, mean, dev),
        ConnectivitySetUpType::ErdosSpectral {
            connectivity,
            spectral_radius,
        } => {
            let mut bernoulli_rng = rand::thread_rng();
            let mut uniform_rng = rand::thread_rng();
            let bernoulli_distr = Bernoulli::new(connectivity).unwrap();
            let erdos_iter = repeat_with(|| {
                if bernoulli_distr.sample(&mut bernoulli_rng) {
                    uniform_rng.gen_range(-0.5..0.5)
                } else {
                    0.0
                }
            })
            .take(number_of_neurons * number_of_neurons);
            let mut conn_matrix =
                DMatrix::from_iterator(number_of_neurons, number_of_neurons, erdos_iter);
            let current_spectral_radius = conn_matrix
                .complex_eigenvalues()
                .iter()
                .fold(0.0, |acc, e| if e.abs() > acc { e.abs() } else { acc });
            let spectral_radius_scale = current_spectral_radius / spectral_radius;
            conn_matrix /= spectral_radius_scale;
            conn_matrix
        }
    }
}

pub fn input_vector(number_of_neurons: usize, input_matrix_setup: InputMatrixSetUp) -> Vec<f64> {
    match input_matrix_setup {
        InputMatrixSetUp::AllConnected => vec![1.0; number_of_neurons],
        InputMatrixSetUp::PercentageConnected { connectivity } => {
            let mut bernoulli_rng = rand::thread_rng();
            let bernoulli_distr = Bernoulli::new(connectivity).unwrap();
            repeat_with(|| {
                if bernoulli_distr.sample(&mut bernoulli_rng) {
                    1.0
                } else {
                    0.0
                }
            })
            .take(number_of_neurons)
            .collect()
        }
    }
}

impl IzhikevichModel {
    pub fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        dt: f64,
        number_of_neurons: usize,                  // won't need it! maybe
        spike_value: f64,                          // won't need it!
        connectivity_setup: ConnectivitySetUpType, // connectivity to the input
        network_init: NetworkInit,                 // the initial states of the network
        input_matrix_setup: InputMatrixSetUp,      // the input of the matrix
        thalmic_input: ThalmicInput,               // thalmic input
    ) -> Self {
        let connectivity_matrix = connectivity_matrix(number_of_neurons, connectivity_setup);
        let input_vector = input_vector(number_of_neurons, input_matrix_setup);
        Self {
            a,
            b,
            c,
            d,
            dt,
            number_of_neurons,
            spike_trashhold: spike_value,
            connectivity_matrix,
            input_vector,
            network_initialization: network_init,
            thalmic_input,
        }
    }

    pub fn input_current(&self, input: InputStep) -> DVector<f64> {
        let input_size = input.input_size();
        let row_vectors_iter = self
            .input_vector
            .iter()
            .map(|x| *x)
            .cycle()
            .take(self.number_of_neurons * input_size);
        let input_matrix =
            DMatrix::from_iterator(self.number_of_neurons, input_size, row_vectors_iter);

        input_matrix * input.vals.clone()
    }

    fn integrate_single_time_step(
        &self,
        input: InputStep,
        current_time: &mut f64,
        mut current_state: IzhikevichModelState,
        times: &mut Vec<f64>,
        states: &mut Vec<IzhikevichModelState>,
    ) -> IzhikevichModelState {
        let time_to_stop = *current_time + input.duration;
        let input_current = self.input_current(input);
        while *current_time <= time_to_stop - self.dt {
            times.push(*current_time);
            states.push(current_state.clone());
            self.euler_step(&mut current_state, &input_current);
            *current_time += self.dt;
        }
        current_state
    }

    pub fn get_states(&self, inputs: Vec<InputStep>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut current_time = 0.0;
        let mut times: Vec<f64> = vec![];
        let mut states: Vec<IzhikevichModelState> = vec![];
        let mut current_state = self.init_state();
        for input in inputs {
            current_state = self.integrate_single_time_step(
                input,
                &mut current_time,
                current_state,
                &mut times,
                &mut states,
            );
        }

        let mut neuron_voltages = repeat_with(|| vec![])
            .take(self.number_of_neurons)
            .collect::<Vec<Vec<f64>>>();

        states.iter().for_each(|state| {
            for neuron in 0..self.number_of_neurons {
                neuron_voltages[neuron].push(state.membrane_potentials[neuron]);
            }
        });

        (times, neuron_voltages)
    }

    pub fn init_state(&self) -> IzhikevichModelState {
        match self.network_initialization {
            NetworkInit::NoRandomWeight {
                membrane_potential,
                recovery_variable,
            } => {
                let membrane_potentials = DVector::from_iterator(
                    self.number_of_neurons,
                    repeat_with(|| membrane_potential).take(self.number_of_neurons),
                );
                let membrane_recovery_variables = DVector::from_iterator(
                    self.number_of_neurons,
                    repeat_with(|| recovery_variable).take(self.number_of_neurons),
                );
                IzhikevichModelState {
                    membrane_potentials,
                    membrane_recovery_variables,
                }
            }
            NetworkInit::NormalWeight {
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
                let membrane_potentials = DVector::from_iterator(
                    self.number_of_neurons,
                    repeat_with(|| membrane_potential_distribution.sample(&mut membrane_rng))
                        .take(self.number_of_neurons),
                );
                let membrane_recovery_variables = DVector::from_iterator(
                    self.number_of_neurons,
                    repeat_with(|| recovery_variable_distribution.sample(&mut recovery_rng))
                        .take(self.number_of_neurons),
                );

                IzhikevichModelState {
                    membrane_potentials,
                    membrane_recovery_variables,
                }
            }
        }
    }
}

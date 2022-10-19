use std::iter::repeat_with;

use crate::integration::{euler_integrate_step, ModelIntegrator};
use either::Either;
use nalgebra::{DMatrix, DVector};
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::Normal;

// pub fn save(times: &Vec<f64>, states: &Vec<IzikevichModelState>, filename: &Path) {
//     let file = match File::create(filename) {
//         Err(e) => {
//             log::error!("Could not open file. Error: {:?}", e);
//             return;
//         }
//         Ok(buf) => buf,
//     };
//     let mut buf = BufWriter::new(file);
//     for (i, state) in states.iter().enumerate() {
//         buf.write_fmt(format_args!("{}", times[i])).unwrap();
//         for val in state.iter() {
//             buf.write_fmt(format_args!(", {}", val)).unwrap();
//         }
//         buf.write_fmt(format_args!("\n")).unwrap();
//     }
//     if let Err(e) = buf.flush() {
//         log::error!("Could not write to file. Error: {:?}", e);
//     }
// }
//
// pub fn save_two(times: &Vec<f64>, states: &Vec<IzikevichModelStateTwo>, filename: &Path) {
//     let file = match File::create(filename) {
//         Err(e) => {
//             log::error!("Could not open file. Error: {:?}", e);
//             return;
//         }
//         Ok(buf) => buf,
//     };
//     let mut buf = BufWriter::new(file);
//     for (i, state) in states.iter().enumerate() {
//         buf.write_fmt(format_args!("{}", times[i])).unwrap();
//         for val in state.membrane_potentials.iter() {
//             buf.write_fmt(format_args!(", {}", val)).unwrap();
//         }
//         buf.write_fmt(format_args!("\n")).unwrap();
//     }
//     if let Err(e) = buf.flush() {
//         log::error!("Could not write to file. Error: {:?}", e);
//     }
// }

pub type IzikevichModelStateChemical = DVector<f64>;

#[derive(Debug, Clone)]
pub struct IzikevichModelState {
    membrane_potentials: DVector<f64>,
    membrane_recovery_variables: DVector<f64>,
}

impl IzikevichModelState {
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

fn create_connectivity_matrix(
    connectivity_graph_type: &ConnectivitySetUpType,
    size: usize,
) -> DMatrix<f64> {
    match connectivity_graph_type {
        ConnectivitySetUpType::ErdosZeroOne(connectivity) => {
            create_erdos_uniform_connectivity_matrix(size, *connectivity, 0.0, 1.0)
        }
        ConnectivitySetUpType::ErdosZeroUp {
            connectivity,
            upper,
        } => create_erdos_uniform_connectivity_matrix(size, *connectivity, 0.0, *upper),
        ConnectivitySetUpType::ErdosLowerUpper {
            connectivity,
            lower,
            upper,
        } => create_erdos_uniform_connectivity_matrix(size, *connectivity, *lower, *upper),
        ConnectivitySetUpType::ErdosNormal {
            connectivity,
            mean,
            dev,
        } => create_erdos_normal_connectivity_matrix(size, *connectivity, *mean, *dev),
    }
}

#[derive(Debug, Clone)]
pub struct InputStep {
    duration: f64,
    vals: DVector<f64>, // now this is going to be the hardest one yet!
}

impl InputStep {
    fn duration(&self) -> f64 {
        self.duration
    }

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
pub struct IzikevichModel {
    pub a: f64,
    pub b: f64,
    pub c: f64, // reset variable
    pub d: f64, // adjust reset variable
    pub number_of_neurons: usize,
    pub spike_trashhold: f64,
    pub dt: f64,
    connectivity_setup: Either<ConnectivitySetUpType, DMatrix<f64>>,
    network_initialization: InitialNetworkStateInit,
    input_matrix_setup: InputMatrixSetUp,
}

impl IzikevichModel {
    pub fn new(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        dt: f64,
        number_of_neurons: usize,
        spike_value: f64,
        connectivity_setup: Either<ConnectivitySetUpType, DMatrix<f64>>,
        network_initialization: InitialNetworkStateInit,
        input_matrix_setup: InputMatrixSetUp,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            dt,
            number_of_neurons,
            spike_trashhold: spike_value,
            connectivity_setup,
            network_initialization,
            input_matrix_setup,
        }
    }

    pub fn connectivity_matrix(&self) -> DMatrix<f64> {
        match &self.connectivity_setup {
            Either::Left(connectivity_graph_type) => {
                create_connectivity_matrix(&connectivity_graph_type, self.number_of_neurons)
            }
            Either::Right(connectivity_matrix) => connectivity_matrix.clone(),
        }
    }

    pub fn input_current(&self, input: InputStep) -> DVector<f64> {
        let input_size = input.input_size();
        let input_matrix = match self.input_matrix_setup {
            InputMatrixSetUp::AllConnected => {
                DMatrix::zeros(self.number_of_neurons as usize, input_size).add_scalar(1.0)
            }
        };
        input_matrix * input.vals.clone()
    }

    pub fn create_model_integrator(
        &self,
        input: InputStep,
        connectivity_matrix: &DMatrix<f64>,
    ) -> ModelIntegrator {
        let input_size = input.input_size();
        let input_matrix = match self.input_matrix_setup {
            InputMatrixSetUp::AllConnected => {
                DMatrix::zeros(self.number_of_neurons as usize, input_size).add_scalar(1.0)
            }
        };
        let current_input = input_matrix * input.vals.clone();
        ModelIntegrator::new(self, &current_input, connectivity_matrix)
    }

    fn integrate_single_time_step(
        &self,
        input: InputStep,
        current_time: &mut f64,
        mut current_state: IzikevichModelState,
        times: &mut Vec<f64>,
        states: &mut Vec<IzikevichModelState>,
        connectivity_matrix: &DMatrix<f64>,
    ) -> IzikevichModelState {
        let time_to_stop = *current_time + input.duration;
        let input_current = self.input_current(input);
        while *current_time <= time_to_stop - self.dt {
            times.push(*current_time);
            states.push(current_state.clone());
            self.euler_step(&mut current_state, &input_current, connectivity_matrix);
            *current_time += self.dt;
        }
        current_state
    }

    fn integrate_single_time_step_chemical(
        &self,
        input: InputStep,
        current_time: &mut f64,
        mut current_state: IzikevichModelStateChemical,
        times: &mut Vec<f64>,
        states: &mut Vec<IzikevichModelStateChemical>,
        connectivity_matrix: &DMatrix<f64>,
    ) -> Option<IzikevichModelStateChemical> {
        let time_to_stop = *current_time + input.duration;
        let model_integrator = self.create_model_integrator(input, &connectivity_matrix);
        while *current_time <= time_to_stop - self.dt {
            times.push(*current_time);
            states.push(current_state.clone());
            euler_integrate_step(
                &model_integrator,
                &mut current_state,
                self.dt,
                self.c,
                self.d,
            );
            *current_time += self.dt;
        }
        Some(current_state)
    }

    pub fn get_states_chemical(&self, inputs: Vec<InputStep>) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut current_time = 0.0;
        let mut times: Vec<f64> = vec![];
        let mut states: Vec<IzikevichModelStateChemical> = vec![];
        let mut current_state = self.init_state_chemical();
        let connectivity_matrix = self.connectivity_matrix();
        for input in inputs {
            current_state = self.integrate_single_time_step_chemical(
                input,
                &mut current_time,
                current_state,
                &mut times,
                &mut states,
                &connectivity_matrix,
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

        // let path = Path::new("./outputs/izikevich_model.dat");
        // save(&times, &states, path);

        Some((times, neuron_voltages))
    }

    pub fn get_states(&self, inputs: Vec<InputStep>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut current_time = 0.0;
        let mut times: Vec<f64> = vec![];
        let mut states: Vec<IzikevichModelState> = vec![];
        let mut current_state = self.init_state();
        let connectivity_matrix = self.connectivity_matrix();
        for input in inputs {
            current_state = self.integrate_single_time_step(
                input,
                &mut current_time,
                current_state,
                &mut times,
                &mut states,
                &connectivity_matrix,
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

        // let path = Path::new("./outputs/izikevich_model.dat");
        // save_two(&times, &states, path);

        (times, neuron_voltages)
    }

    pub fn init_state_chemical(&self) -> IzikevichModelStateChemical {
        match self.network_initialization {
            InitialNetworkStateInit::NoRandomWeight {
                membrane_potential,
                recovery_variable,
            } => {
                let neuron_iter = repeat_with(|| membrane_potential)
                    .take(self.number_of_neurons)
                    .chain(repeat_with(|| recovery_variable).take(self.number_of_neurons));
                IzikevichModelStateChemical::from_iterator(2 * self.number_of_neurons, neuron_iter)
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
                IzikevichModelStateChemical::from_iterator(2 * self.number_of_neurons, neuron_iter)
            }
        }
    }

    pub fn init_state(&self) -> IzikevichModelState {
        match self.network_initialization {
            InitialNetworkStateInit::NoRandomWeight {
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
                IzikevichModelState {
                    membrane_potentials,
                    membrane_recovery_variables,
                }
                // IzikevichModelState::from_iterator(2 * self.number_of_neurons, neuron_iter)
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

                IzikevichModelState {
                    membrane_potentials,
                    membrane_recovery_variables,
                }
            }
        }
    }

    pub fn handle_post_fire(&self, post_fire: &mut IzikevichModelStateChemical) {
        for n_index in 0..self.number_of_neurons {
            if post_fire[n_index] > self.spike_trashhold {
                post_fire[n_index] = self.c;
                post_fire[n_index + self.number_of_neurons] =
                    post_fire[n_index + self.number_of_neurons] + self.d;
            }
        }
    }
}

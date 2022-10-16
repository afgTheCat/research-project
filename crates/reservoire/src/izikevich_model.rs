use crate::integration::ModelIntegrator;
use either::Either;
use nalgebra::{DMatrix, DVector};
use ode_solvers::Dopri5;
use rand::{distributions::Bernoulli, prelude::Distribution, Rng};
use rand_distr::Normal;
use std::{
    fs::File,
    io::{BufWriter, Write},
    iter::repeat_with,
    path::Path,
};

pub fn save(times: &Vec<f64>, states: &Vec<IzikevichModelState>, filename: &Path) {
    let file = match File::create(filename) {
        Err(e) => {
            log::error!("Could not open file. Error: {:?}", e);
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
        log::error!("Could not write to file. Error: {:?}", e);
    }
}

pub type IzikevichModelState = DVector<f64>;

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
    Erdos { connectivity: f64 },
}

fn create_random_graph(
    connectivity_graph_type: &ConnectivitySetUpType,
    size: usize,
) -> DMatrix<f64> {
    match connectivity_graph_type {
        ConnectivitySetUpType::Erdos { connectivity } => {
            let mut rng = rand::thread_rng();
            let bernoulli_distr = Bernoulli::new(*connectivity).unwrap();
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
    vals: DVector<f64>,
}

impl InputStep {
    fn duration(&self) -> f64 {
        self.duration
    }

    fn vals(&self) -> &DVector<f64> {
        &self.vals
    }

    fn input_size(&self) -> usize {
        self.vals.len()
    }

    pub fn new(duration: f64, vals: DVector<f64>) -> Self {
        Self { duration, vals }
    }
}

#[derive(Debug, Clone)]
pub struct IzikevichModel {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub number_of_neurons: usize,
    pub spike_value: f64,
    dt: f64,
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
            spike_value,
            connectivity_setup,
            network_initialization,
            input_matrix_setup,
        }
    }

    pub fn connectivity_matrix(&self) -> DMatrix<f64> {
        match &self.connectivity_setup {
            Either::Left(connectivity_graph_type) => {
                create_random_graph(&connectivity_graph_type, self.number_of_neurons)
            }
            Either::Right(connectivity_matrix) => connectivity_matrix.clone(),
        }
    }

    // we have to create the input I and the
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
        let current_input = input_matrix * input.vals;
        log::info!("current input: {:#x?}", current_input);
        ModelIntegrator::new(self, &current_input, connectivity_matrix)
    }

    fn integrate_single_state(
        &self,
        input: InputStep,
        mut current_time: f64,
        mut current_state: IzikevichModelState,
        times: &mut Vec<f64>,
        states: &mut Vec<IzikevichModelState>,
        connectivity_matrix: &DMatrix<f64>,
    ) -> Option<IzikevichModelState> {
        let time_to_stop = current_time + input.duration;
        let model_integrator = self.create_model_integrator(input, &connectivity_matrix);
        loop {
            let mut stepper = Dopri5::new(
                model_integrator.clone(),
                current_time,
                time_to_stop,
                self.dt,
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
            if current_time > time_to_stop - self.dt {
                break Some(current_state);
            }
        }
    }

    pub fn get_states(&self, inputs: Vec<InputStep>) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let current_time = 0.0;
        let mut times: Vec<f64> = vec![];
        let mut states: Vec<IzikevichModelState> = vec![];
        let mut current_state = self.init_state();
        let connectivity_matrix = self.connectivity_matrix();
        for input in inputs {
            current_state = self.integrate_single_state(
                input,
                current_time,
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

        Some((times, neuron_voltages))
    }

    pub fn get_states_with_val(&self) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let input_vals = DVector::from_vec(vec![10.0]);
        let inputs = vec![InputStep::new(1000.0, input_vals)];
        self.get_states(inputs)
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

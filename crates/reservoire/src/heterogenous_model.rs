use rand_distr::{Bernoulli, Distribution};
use std::{collections::HashMap, ops::Range};

type NeuronId = usize;
type InputId = usize;

#[derive(Clone)]
pub struct NetworkState {
    time: f64,
    firing_neurons: Vec<NeuronId>,
    membrane_potentials: Vec<f64>,
}

impl NetworkState {
    fn new(time: f64, firing_neurons: Vec<NeuronId>, membrane_potentials: Vec<f64>) -> Self {
        Self {
            time,
            firing_neurons,
            membrane_potentials,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuronParameters {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl NeuronParameters {
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }
}

impl Default for NeuronParameters {
    fn default() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }
}

/// An Izikevich neuron with it's parameters and inner state
#[derive(Debug, Clone)]
pub struct Neuron {
    pub v: f64,
    pub u: f64,
    parameters: NeuronParameters,
}

impl Neuron {
    fn new(parameters: NeuronParameters) -> Self {
        Self {
            v: parameters.c,
            u: parameters.b * parameters.c,
            parameters,
        }
    }

    pub fn integrate(&mut self, input: f64, dt: f64) -> bool {
        // Izikevich model
        self.v += dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input);
        self.u += dt * self.parameters.a * (self.parameters.b * self.v - self.u);

        if self.v >= 35.0 {
            self.v = self.parameters.c;
            self.u += self.parameters.d;

            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
struct SynapticConnection {
    target: NeuronId,
    weight: f64,
}

impl SynapticConnection {
    fn new(target: NeuronId, weight: f64) -> Self {
        Self { target, weight }
    }

    fn target(&self) -> usize {
        self.target
    }

    fn weight(&self) -> f64 {
        self.weight
    }
}

#[derive(Debug, Clone)]
pub struct InputStepHeterogenous {
    duration: f64,
    input: Vec<f64>, // should contain all neurons
}

impl InputStepHeterogenous {
    pub fn new(duration: f64, input: Vec<f64>) -> Self {
        Self { duration, input }
    }

    fn duration(&self) -> f64 {
        self.duration
    }

    pub fn input(&self) -> &[f64] {
        self.input.as_ref()
    }
}

struct NetworkParameters {
    pub input_layer_connections: HashMap<InputId, Vec<SynapticConnection>>,
    pub input_layer_connectivity: f64,
    pub dt: f64,
}

impl NetworkParameters {
    fn new(dt: f64) -> Self {
        Self {
            dt,
            input_layer_connectivity: 0.1,
            input_layer_connections: HashMap::new(),
        }
    }

    fn push_input(&mut self, input_id: InputId, connection: SynapticConnection) {
        self.input_layer_connections
            .entry(input_id)
            .and_modify(|cv| cv.push(connection.clone()))
            .or_insert(vec![connection]);
    }
}

pub struct HeterogenousReserviore {
    network_parameters: NetworkParameters,

    // the neurons in the network
    neurons: Vec<Neuron>,

    // neuron and what it connects to
    connections: HashMap<NeuronId, Vec<SynapticConnection>>,
}

impl HeterogenousReserviore {
    fn number_of_neurons(&self) -> usize {
        self.neurons.len()
    }

    fn neuron_ids(&self) -> Range<NeuronId> {
        0..self.number_of_neurons()
    }

    fn neuron_targeting(&self, neuron: NeuronId) -> impl Iterator<Item = &SynapticConnection> {
        self.connections
            .get(&neuron)
            .and_then(|conn| Some(conn.iter()))
            .or_else(|| Some([].iter()))
            .unwrap()
    }

    fn neurons_firing(&mut self, input: Vec<f64>) -> Vec<NeuronId> {
        input
            .into_iter()
            .enumerate()
            .filter_map(|(neuron_id, input)| {
                if self.neurons[neuron_id].integrate(input, self.network_parameters.dt) {
                    Some(neuron_id)
                } else {
                    None
                }
            })
            .collect()
    }

    fn create_input(&self, input_step: InputStepHeterogenous) -> Vec<f64> {
        let mut inputs = vec![0.0; self.number_of_neurons()];
        for (input_id, input) in input_step.input().into_iter().enumerate() {
            if let Some(input_connections) = self
                .network_parameters
                .input_layer_connections
                .get(&input_id)
            {
                for connection in input_connections {
                    let target = connection.target();
                    let weight = connection.weight();

                    inputs[target] += input * weight
                }
            }
        }
        inputs
    }

    fn integrate_network_input_step(
        &mut self,
        input_step: InputStepHeterogenous,
        mut time: f64,
        mut firing_neurons: Vec<NeuronId>,
        state_collector: &mut Vec<NetworkState>,
    ) -> (f64, Vec<NeuronId>) {
        let mut input_step_firings: Vec<NetworkState> = vec![];
        let end_time = time + input_step.duration();
        let mut input = self.create_input(input_step);

        while time < end_time {
            for neuron_id in self.neuron_ids() {
                if firing_neurons.contains(&neuron_id) {
                    for synaptic_conn in self.neuron_targeting(neuron_id) {
                        let target = synaptic_conn.target();
                        let weight = synaptic_conn.weight();

                        input[target] += weight;
                    }
                }
            }

            firing_neurons = self.neurons_firing(input);
            input_step_firings.push(NetworkState::new(time, firing_neurons.clone(), vec![]));

            input = vec![0_f64; self.number_of_neurons()];
            time += self.network_parameters.dt;
        }

        state_collector.extend_from_slice(&input_step_firings);
        (time, firing_neurons)
    }

    pub fn setup_input_connections(&mut self, inputs: &[InputStepHeterogenous]) {
        self.network_parameters.input_layer_connections = HashMap::new();
        let input_len = inputs[0].input().len();
        let mut bernoulli_rng = rand::thread_rng();
        let bernoulli_distr =
            Bernoulli::new(self.network_parameters.input_layer_connectivity).unwrap();
        for input in 0..input_len {
            for neuron in 0..self.number_of_neurons() {
                if bernoulli_distr.sample(&mut bernoulli_rng) {
                    self.network_parameters
                        .push_input(input, SynapticConnection::new(neuron, 1.0))
                }
            }
        }
    }

    // integrates the network, returns the time, the firing neurons and the states
    pub fn integrate_network(&mut self, inputs: Vec<InputStepHeterogenous>) -> Vec<NetworkState> {
        // we should add the input to the firings
        let mut state_collector: Vec<NetworkState> = vec![];
        let mut firing_neurons: Vec<NeuronId> = vec![];
        self.setup_input_connections(&inputs);

        let mut time = 0_f64;
        for input in inputs {
            (time, firing_neurons) =
                self.integrate_network_input_step(input, time, firing_neurons, &mut state_collector)
        }
        state_collector
    }

    // will need more paramters
    pub fn new(number_of_neurons: usize, dt: f64) -> Self {
        let network_parameters = NetworkParameters::new(dt);
        let neurons = vec![Neuron::new(NeuronParameters::default()); number_of_neurons];
        let connections = (0..number_of_neurons)
            .map(|neuron_id| {
                let connections = (0..number_of_neurons)
                    .filter_map(|target| {
                        if target != neuron_id {
                            Some(SynapticConnection::new(target, 1.0))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                (neuron_id, connections)
            })
            .collect::<HashMap<_, _>>();
        Self {
            network_parameters,
            neurons,
            connections,
        }
    }
}

use crate::izikevich_model::ConnectivitySetUpType;
use std::collections::{HashMap, HashSet};

const DT: f64 = 0.05;

struct Connection {
    target: usize,
    weight: f64,
    delay: usize,
}

pub struct Parameters {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl Parameters {
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    fn c(&self) -> f64 {
        self.c
    }

    fn b(&self) -> f64 {
        self.b
    }

    fn a(&self) -> f64 {
        self.a
    }

    fn d(&self) -> f64 {
        self.d
    }
}

struct Neuron {
    // The current state of the neuron
    v: f64,
    u: f64,
    // The input current to the neuron
    input: f64,
    // The parameters of the neuron
    parameters: Parameters,
}

impl Neuron {
    fn new(parameters: Parameters) -> Self {
        Self {
            v: parameters.c,
            u: parameters.b * parameters.c,
            input: 0.0,
            // incoming,
            // outgoing,
            parameters,
        }
    }

    fn a(&self) -> f64 {
        self.parameters.a()
    }

    fn b(&self) -> f64 {
        self.parameters.b()
    }

    fn c(&self) -> f64 {
        self.parameters.c()
    }

    fn d(&self) -> f64 {
        self.parameters.d()
    }

    fn integrate(&mut self) {
        // Izikevich model
        self.v += DT * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + self.input);
        self.u += DT * self.a() * (self.b() * self.v - self.u);
    }

    // fires and deplete it
    fn fires(&mut self) -> bool {
        if self.v >= 30.0 {
            self.v = self.c();
            self.u += self.d();
            true
        } else {
            false
        }
    }

    fn adjust_v(&mut self, input: f64) {
        self.v += input
    }
}

struct NeuronWithConnection {
    neuron: Neuron,
    connection: Vec<ConnectionTarget>,
}

impl NeuronWithConnection {
    fn new(neuron: Neuron, connection: Vec<ConnectionTarget>) -> Self {
        Self { neuron, connection }
    }

    fn integrate(&mut self) {
        self.neuron.integrate()
    }

    fn fire(&mut self) -> Option<&Vec<ConnectionTarget>> {
        if self.neuron.fires() {
            Some(&self.connection)
        } else {
            None
        }
    }

    fn adjust_v(&mut self, input: f64) {
        self.neuron.adjust_v(input)
    }
}

#[derive(Debug, Clone)]
pub struct Input {
    duration: f64,
    vals: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConnectionTarget {
    weight: f64,
    target: usize,
}

impl ConnectionTarget {
    pub fn new(weight: f64, target: usize) -> Self {
        Self { weight, target }
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn target(&self) -> usize {
        self.target
    }
}

type ConnectionGraph = HashMap<usize, Vec<ConnectionTarget>>;

pub struct Network {
    neurons: HashMap<usize, NeuronWithConnection>,
}

impl Network {
    pub fn new(parameters: Vec<Parameters>, connectivity_setup: ConnectivitySetUpType) {
        let mut neurons_with_con = HashMap::new();
        let number_of_neurons = parameters.len();
        let connections = connectivity_setup.connections(number_of_neurons);

        for (id, p) in parameters.into_iter().enumerate() {
            let neuron = Neuron::new(p);
            let connections = connections.get(&id).cloned().unwrap_or(vec![]);
            let neuron_with_con = NeuronWithConnection::new(neuron, connections);
            neurons_with_con.insert(id, neuron_with_con);
        }
    }

    fn integrate(&mut self) {
        for neuron in self.neurons.values_mut() {
            neuron.integrate()
        }
    }

    fn fire(&mut self) {
        let spikes = self
            .neurons
            .values_mut()
            .filter_map(|neuron| {
                neuron.fire().and_then(|connections| {
                    let connections = connections
                        .iter()
                        .map(|connection| (connection.target, connection.weight))
                        .collect::<Vec<_>>();
                    Some(connections)
                })
            })
            .flatten()
            .collect::<Vec<_>>();

        for (dest, weight) in spikes {
            // unwrap should never fail
            self.neurons.get_mut(&dest).unwrap().adjust_v(weight)
        }
    }

    fn integrate_and_fire(&mut self, num_step: u64) {
        let mut t = 0_f64;

        for _ in 0..num_step {
            t += DT;
            self.integrate();
            self.fire();
        }
    }
}

#[cfg(test)]
mod test {}

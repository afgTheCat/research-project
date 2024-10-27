use nalgebra::{DMatrix, DVector};
use std::f64;

struct Model {
    pub a: DVector<f64>,
    pub b: DVector<f64>,
    pub c: DVector<f64>,
    pub d: DVector<f64>,
    pub v: DVector<f64>,
    pub u: DVector<f64>,
    pub connections: DMatrix<f64>,
}

impl Model {
    fn new(
        a: DVector<f64>,
        b: DVector<f64>,
        c: DVector<f64>,
        d: DVector<f64>,
        v: DVector<f64>,
        u: DVector<f64>,
        connections: DMatrix<f64>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            v,
            u,
            connections,
        }
    }

    // dt is in miliseconds
    fn step(&mut self, mut input: DVector<f64>, dt: f64) {
        let firings = self
            .v
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| (*v > 30.).then(|| i))
            .collect::<Vec<_>>();

        for i in firings {
            self.v[i] = self.c[i];
            self.u[i] = self.u[i] + self.d[i];
            input += self.connections.column(i);
        }

        self.v +=
            dt * ((0.04 * &self.v * &self.v + 5. * &self.v - &self.u + input).add_scalar(140.));
        self.u += dt * &self.a * (&self.b * &self.v - &self.u);
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{DMatrix, DVector};

    use super::Model;

    #[test]
    fn model_test() {
        let a = DVector::from_vec(vec![0.02]);
        let b = DVector::from_vec(vec![0.2]);
        let c = DVector::from_vec(vec![-65.]);
        let d = DVector::from_vec(vec![8.]);
        let v = DVector::from_vec(vec![-65.]);
        let u = DVector::from_vec(vec![0.2 * -64.]);
        let connections = DMatrix::from_vec(1, 1, vec![0.]);
        let input = DVector::from_vec(vec![15.]);

        let mut model = Model::new(a, b, c, d, v, u, connections);
        let mut voltages = vec![];

        for _ in 0..10000 {
            model.step(input.clone(), 1.);
            voltages.push(model.v[0]);
        }
    }
}

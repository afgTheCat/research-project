use nalgebra::{DMatrix, DVector};
use std::f64;

// TODO: multi reservoire setup
pub struct NewModel {
    pub a: DVector<f64>,
    pub b: DVector<f64>,
    pub c: DVector<f64>,
    pub d: DVector<f64>,
    pub v: DVector<f64>,
    pub u: DVector<f64>,
    pub connections: DMatrix<f64>,
}

impl NewModel {
    pub fn new(
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

    // we basically have to steps: diffuse and excite
    pub fn diffuse(&mut self, mut input: DVector<f64>) -> DVector<f64> {
        let firings = self
            .v
            .into_iter()
            .enumerate()
            .filter(|(_, v)| (**v > 30.))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        for i in firings {
            self.v[i] = self.c[i];
            self.u[i] += self.d[i];
            input += self.connections.column(i);
        }
        input
    }

    pub fn excite(&mut self, input: DVector<f64>, dt: f64) -> DVector<f64> {
        self.v +=
            dt * ((0.04 * &self.v * &self.v + 5. * &self.v - &self.u + input).add_scalar(140.));
        self.u += dt * &self.a * (&self.b * &self.v - &self.u);
        self.v.clone()
    }
}

#[cfg(test)]
mod test {
    use super::NewModel;
    use nalgebra::{DMatrix, DVector};
    use std::fs::File;

    const DATA_LOCATION: &str = "../../data/output.csv";

    #[test]
    fn model_test() {
        let data = File::create(DATA_LOCATION).unwrap();
        let mut wtr = csv::Writer::from_writer(data);

        let a = DVector::from_vec(vec![0.02]);
        let b = DVector::from_vec(vec![0.2]);
        let c = DVector::from_vec(vec![-65.]);
        let d = DVector::from_vec(vec![8.]);
        let v = DVector::from_vec(vec![-65.]);
        let u = DVector::from_vec(vec![0.2 * -64.]);
        let connections = DMatrix::from_vec(1, 1, vec![0.]);
        let input = DVector::from_vec(vec![15.]);

        let mut model = NewModel::new(a, b, c, d, v, u, connections);
        let mut voltages = vec![];

        // in miliseconds
        for t in 0..500 {
            let excited_input = model.diffuse(input.clone());
            let voltage = model.excite(excited_input, 1.);
            voltages.push([t.to_string(), voltage[0].to_string()]);
        }

        for rec in voltages {
            wtr.write_record(rec).unwrap();
        }

        wtr.flush().unwrap();
    }
}

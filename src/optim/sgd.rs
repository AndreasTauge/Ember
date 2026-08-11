use crate::autograd::tensor_autodiff::{Graph, TensorId};

pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Sgd {
        Self { learning_rate }
    }

    pub fn step(&self, graph: &mut Graph, parameters: &[TensorId]) {
        for &parameter in parameters {
            let change = graph.grad(parameter).mul_scalar(-self.learning_rate);
            graph.add_to_data(parameter, &change);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn step_updates_only_the_supplied_parameters() {
        let mut graph = Graph::new();
        let parameter = graph.tensor(Tensor::new(vec![2.0, -1.0], vec![2]));
        let untouched = graph.tensor(Tensor::new(vec![4.0, 5.0], vec![2]));
        let loss = graph.sum(parameter);
        graph.backward(loss);

        Sgd::new(0.1).step(&mut graph, &[parameter]);

        assert_eq!(graph.data(parameter).values(), &[1.9, -1.1]);
        assert_eq!(graph.data(untouched).values(), &[4.0, 5.0]);
    }
}

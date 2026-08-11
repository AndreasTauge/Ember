use crate::{
    autograd::tensor_autodiff::{Graph, TensorId},
    tensor::Tensor,
};

pub struct Linear {
    pub weight: TensorId,
    pub bias: TensorId,
}

impl Linear {
    pub fn new(graph: &mut Graph, weight: Tensor, bias: Tensor) -> Linear {
        let weight = graph.tensor(weight);
        let bias = graph.tensor(bias);
        Linear { weight, bias }
    }

    pub fn parameters(&self) -> [TensorId; 2] {
        [self.weight, self.bias]
    }

    pub fn forward(&self, graph: &mut Graph, input: TensorId) -> TensorId {
        let weighted = graph.matmul(input, self.weight);
        graph.add(weighted, self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn forward_broadcasts_bias_and_backward_reduces_its_gradient() {
        let mut graph = Graph::new();
        let linear = Linear::new(
            &mut graph,
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::new(vec![0.5, -0.5], vec![2]),
        );
        let input = graph.tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));

        let output = linear.forward(&mut graph, input);
        assert_eq!(graph.data(output).shape(), &[2, 2]);
        assert_close(graph.data(output).values(), &[7.5, 9.5, 15.5, 21.5]);

        let loss = graph.sum(output);
        graph.backward(loss);

        assert_close(graph.grad(input).values(), &[3.0, 7.0, 3.0, 7.0]);
        assert_close(graph.grad(linear.weight).values(), &[4.0, 4.0, 6.0, 6.0]);
        assert_close(graph.grad(linear.bias).values(), &[2.0, 2.0]);
    }

    #[test]
    fn parameters_returns_weight_and_bias() {
        let mut graph = Graph::new();
        let linear = Linear::new(
            &mut graph,
            Tensor::new(vec![1.0], vec![1, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        assert_eq!(linear.parameters(), [linear.weight, linear.bias]);
    }
}

use crate::{
    autograd::tensor_autodiff::{Graph, TensorId},
    tensor::Tensor,
};

pub struct Linear {
    pub weight: TensorId,
}

impl Linear {
    pub fn new(graph: &mut Graph, weight: Tensor) -> Linear {
        let weight = graph.tensor(weight);
        Linear { weight }
    }

    pub fn forward(&self, graph: &mut Graph, input: TensorId) -> TensorId {
        graph.matmul(input, self.weight)
    }
}

use crate::tensor::Tensor;

pub type TensorId = usize;

pub struct Node {
    data: Tensor,
    grad: Tensor,
    operation: Operation,
}

#[derive(Clone, Copy)]
enum Operation {
    Leaf,
    Add(TensorId, TensorId),
    Mul(TensorId, TensorId),
    Sub(TensorId, TensorId),
    Div(TensorId, TensorId),
    Pow(TensorId, f32),
    Exp(TensorId),
    Tanh(TensorId),
    Sum(TensorId),
    MatMul(TensorId, TensorId),
}

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new(nodes: Vec<Node>) -> Graph {
        Graph { nodes }
    }
    pub fn tensor(&mut self, data: Tensor) -> TensorId {
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Leaf,
        });
        id
    }
    pub fn grad(&self, id: TensorId) -> &Tensor {
        &self.nodes[id].grad
    }

    pub fn add(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let data = self.nodes[left].data.add(&self.nodes[right].data);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Add(left, right),
        });
        id
    }

    pub fn mul(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let data = self.nodes[left].data.mul(&self.nodes[right].data);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Mul(left, right),
        });
        id
    }

    pub fn sub(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let data = self.nodes[left].data.sub(&self.nodes[right].data);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Sub(left, right),
        });
        id
    }

    pub fn div(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let data = self.nodes[left].data.div(&self.nodes[right].data);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Div(left, right),
        });
        id
    }

    pub fn pow(&mut self, left: TensorId, exp: f32) -> TensorId {
        let data = self.nodes[left].data.pow(exp);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Pow(left, exp),
        });
        id
    }

    pub fn exp(&mut self, left: TensorId) -> TensorId {
        let data = self.nodes[left].data.exp();
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Exp(left),
        });
        id
    }

    pub fn tanh(&mut self, left: TensorId) -> TensorId {
        let data = self.nodes[left].data.tanh();
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Tanh(left),
        });
        id
    }

    pub fn sum(&mut self, left: TensorId) -> TensorId {
        let data = self.nodes[left].data.sum();
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Sum(left),
        });
        id
    }

    pub fn matmul(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let data = self.nodes[left].data.matmul(&self.nodes[right].data);
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::MatMul(left, right),
        });
        id
    }

    pub fn backward(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        for node in &mut self.nodes {
            node.grad = node.data.zeros_like();
        }

        let end = self.nodes.len() - 1;
        self.nodes[end].grad = self.nodes[end].data.ones_like();

        for i in (0..=end).rev() {
            let grad = self.nodes[i].grad.clone();
            let operation = self.nodes[i].operation;

            match operation {
                Operation::Leaf => {}

                Operation::Add(left, right) => {
                    self.nodes[left].grad = self.nodes[left].grad.add(&grad);
                    self.nodes[right].grad = self.nodes[right].grad.add(&grad);
                }

                Operation::Mul(left, right) => {
                    let left_data = &self.nodes[left].data;
                    let right_data = &self.nodes[right].data;

                    let left_grad = right_data.mul(&grad);
                    let right_grad = left_data.mul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&left_grad);
                    self.nodes[right].grad = self.nodes[right].grad.add(&right_grad);
                }

                Operation::Sub(left, right) => {
                    self.nodes[left].grad = self.nodes[left].grad.add(&grad);
                    self.nodes[right].grad = self.nodes[right].grad.sub(&grad);
                }

                Operation::Div(left, right) => {
                    let left_data = &self.nodes[left].data;
                    let right_data = &self.nodes[right].data;

                    let left_grad = right_data.ones_like().div(right_data).mul(&grad);

                    let right_grad = left_data
                        .div(&right_data.pow(2.0))
                        .mul_scalar(-1.0)
                        .mul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&left_grad);
                    self.nodes[right].grad = self.nodes[right].grad.add(&right_grad);
                }

                Operation::Pow(left, exp) => {
                    let left_data = &self.nodes[left].data;
                    let local_grad = left_data.pow(exp - 1.0).mul_scalar(exp).mul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&local_grad);
                }

                Operation::Exp(left) => {
                    let data = &self.nodes[i].data;
                    let local_grad = data.mul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&local_grad);
                }

                Operation::Tanh(left) => {
                    let data = &self.nodes[i].data;
                    let local_grad = data.pow(2.0).mul_scalar(-1.0).add_scalar(1.0).mul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&local_grad);
                }

                Operation::Sum(value) => {
                    let scalar_grad = grad.item();
                    let local_grad = self.nodes[value].data.full_like(scalar_grad);

                    self.nodes[value].grad = self.nodes[value].grad.add(&local_grad);
                }

                Operation::MatMul(left, right) => {
                    let left_data = &self.nodes[left].data;
                    let right_data = &self.nodes[right].data;

                    let left_grad = grad.matmul(&right_data.transpose());
                    let right_grad = left_data.transpose().matmul(&grad);

                    self.nodes[left].grad = self.nodes[left].grad.add(&left_grad);
                    self.nodes[right].grad = self.nodes[right].grad.add(&right_grad);
                }
            }
        }
    }
}

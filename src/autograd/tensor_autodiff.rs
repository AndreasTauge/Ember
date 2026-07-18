use crate::tensor::Tensor;

type TensorId = usize;

struct Node {
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
}

struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    fn tensor(&mut self, data: Tensor) -> TensorId {
        let id = self.nodes.len();
        let grad = data.zeros_like();

        self.nodes.push(Node {
            data,
            grad,
            operation: Operation::Leaf,
        });
        id
    }

    fn add(&mut self, left: TensorId, right: TensorId) -> TensorId {
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

    fn mul(&mut self, left: TensorId, right: TensorId) -> TensorId {
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

    fn sub(&mut self, left: TensorId, right: TensorId) -> TensorId {
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

    fn div(&mut self, left: TensorId, right: TensorId) -> TensorId {
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

    fn pow(&mut self, left: TensorId, exp: f32) -> TensorId {
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

    fn exp(&mut self, left: TensorId) -> TensorId {
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

    fn tanh(&mut self, left: TensorId) -> TensorId {
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

    fn backward(&mut self) {
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
            }
        }
    }
}

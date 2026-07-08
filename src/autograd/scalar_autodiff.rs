type ValueId = usize;

struct Node {
    data: f32,
    grad: f32,
    operation: Operation,
}

#[derive(Clone, Copy)]
enum Operation {
    Leaf,
    Add(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Div(ValueId, ValueId),
    Pow(ValueId, f32),
    Exp(ValueId),
    Tanh(ValueId),
}

struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    fn value(&mut self, data: f32) -> ValueId {
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Leaf,
        });
        id
    }

    fn add(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let data = self.nodes[left].data + self.nodes[right].data;
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Add(left, right),
        });
        id
    }

    fn mul(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let data = self.nodes[left].data * self.nodes[right].data;
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Mul(left, right),
        });
        id
    }

    fn sub(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let data = self.nodes[left].data - self.nodes[right].data;
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Sub(left, right),
        });
        id
    }

    fn div(&mut self, left: ValueId, right: ValueId) -> ValueId {
        let data = self.nodes[left].data / self.nodes[right].data;
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Div(left, right),
        });
        id
    }

    fn pow(&mut self, left: ValueId, exp: f32) -> ValueId {
        let data = self.nodes[left].data.powf(exp);
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Pow(left, exp),
        });
        id
    }

    fn exp(&mut self, left: ValueId) -> ValueId {
        let data = self.nodes[left].data.exp();
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Exp(left),
        });
        id
    }

    fn tanh(&mut self, left: ValueId) -> ValueId {
        let data = self.nodes[left].data.tanh();
        let id = self.nodes.len();

        self.nodes.push(Node {
            data,
            grad: 0.0,
            operation: Operation::Tanh(left),
        });
        id
    }

    fn backward(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        for node in &mut self.nodes {
            node.grad = 0.0;
        }

        let end = self.nodes.len() - 1;
        self.nodes[end].grad = 1.0;

        for i in (0..=end).rev() {
            let grad = self.nodes[i].grad;
            let operation = self.nodes[i].operation;

            match operation {
                Operation::Leaf => {}

                Operation::Add(left, right) => {
                    self.nodes[left].grad += grad;
                    self.nodes[right].grad += grad;
                }

                Operation::Mul(left, right) => {
                    let left_data = self.nodes[left].data;
                    let right_data = self.nodes[right].data;

                    self.nodes[left].grad += right_data * grad;
                    self.nodes[right].grad += left_data * grad;
                }

                Operation::Sub(left, right) => {
                    self.nodes[left].grad += grad;
                    self.nodes[right].grad -= grad;
                }

                Operation::Div(left, right) => {
                    let left_data = self.nodes[left].data;
                    let right_data = self.nodes[right].data;

                    self.nodes[left].grad += (1.0 / right_data) * grad;
                    self.nodes[right].grad += (-left_data / (right_data.powi(2))) * grad;
                }

                Operation::Pow(left, exp) => {
                    let left_data = self.nodes[left].data;
                    self.nodes[left].grad += exp * left_data.powf(exp - 1.0) * grad;
                }

                Operation::Exp(left) => {
                    let data = self.nodes[i].data;
                    self.nodes[left].grad += data * grad;
                }

                Operation::Tanh(left) => {
                    let data = self.nodes[i].data;
                    self.nodes[left].grad += (1.0 - data.powi(2)) * grad;
                }
            }
        }
    }
}

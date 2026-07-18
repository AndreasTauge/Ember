use ember::autograd::tensor_autodiff::Graph;
use ember::tensor::Tensor;

fn main() {
    let mut graph = Graph::new(vec![]);
    let x = graph.tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));
    let y = graph.mul(x, x);
    let _loss = graph.sum(y);
    graph.backward();
    println!("x grad = {:?}", graph.grad(x).values());
}

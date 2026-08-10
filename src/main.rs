use ember::autograd::tensor_autodiff::Graph;
use ember::tensor::Tensor;

fn main() {
    let mut graph = Graph::new();
    let x = graph.tensor(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    ));
    let y = graph.matmul(x, x);
    graph.backward(y);
    println!("x grad = {:?}", graph.grad(x).values());
}

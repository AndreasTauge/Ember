#[derive(Clone)]
pub struct Tensor {
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(values: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let expected = shape.iter().product();
        assert_eq!(values.len(), expected);

        Tensor { values, shape }
    }

    fn binary_op<F>(&self, other: &Tensor, operation: F) -> Tensor
    where
        F: Fn(f32, f32) -> f32,
    {
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let len = shape.iter().product();

        let values = (0..len)
            .map(|index| {
                let left_index = Self::broadcast_index(index, &shape, &self.shape);

                let right_index = Self::broadcast_index(index, &shape, &other.shape);

                let left = self.values[left_index];
                let right = other.values[right_index];

                operation(left, right)
            })
            .collect();

        Tensor { values, shape }
    }

    fn broadcast_shape(left: &[usize], right: &[usize]) -> Vec<usize> {
        let rank = left.len().max(right.len());
        let mut shape = Vec::with_capacity(rank);

        for axis in 0..rank {
            let left_dim = axis
                .checked_sub(rank - left.len())
                .map_or(1, |index| left[index]);
            let right_dim = axis
                .checked_sub(rank - right.len())
                .map_or(1, |index| right[index]);
            assert!(
                left_dim == right_dim || left_dim == 1 || right_dim == 1,
                "cannot broadcast shapes {:?} and {:?}",
                left,
                right
            );
            shape.push(left_dim.max(right_dim));
        }

        shape
    }

    fn broadcast_index(mut index: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
        let rank_offset = output_shape.len() - input_shape.len();
        let mut input_index = 0;
        let mut input_stride = 1;

        for output_axis in (0..output_shape.len()).rev() {
            let coordinate = index % output_shape[output_axis];
            index /= output_shape[output_axis];

            if output_axis >= rank_offset {
                let input_dim = input_shape[output_axis - rank_offset];
                input_index += if input_dim == 1 { 0 } else { coordinate } * input_stride;
                input_stride *= input_dim;
            }
        }

        input_index
    }

    pub(crate) fn sum_to_shape(&self, shape: &[usize]) -> Tensor {
        let broadcasted = Self::broadcast_shape(&self.shape, shape);
        assert_eq!(
            broadcasted, self.shape,
            "cannot reduce {:?} to {:?}",
            self.shape, shape
        );

        let mut values = vec![0.0; shape.iter().product()];
        for (index, value) in self.values.iter().enumerate() {
            let target_index = Self::broadcast_index(index, &self.shape, shape);
            values[target_index] += value;
        }
        Tensor::new(values, shape.to_vec())
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |left, right| left + right)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |left, right| left - right)
    }
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |left, right| left * right)
    }
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, |left, right| left / right)
    }
    pub fn exp(&self) -> Tensor {
        let data = self.values.iter().map(|x| x.exp()).collect();
        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn pow(&self, exponent: f32) -> Tensor {
        let data = self.values.iter().map(|x| x.powf(exponent)).collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn tanh(&self) -> Tensor {
        let data = self.values.iter().map(|x| x.tanh()).collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn sum(&self) -> Tensor {
        let data = self.values.iter().sum();

        Tensor {
            values: vec![data],
            shape: vec![],
        }
    }
    pub fn mean(&self) -> Tensor {
        let data = self.values.iter().sum::<f32>() / self.values.len() as f32;

        Tensor {
            values: vec![data],
            shape: vec![],
        }
    }
    pub fn add_scalar(&self, value: f32) -> Tensor {
        let data = self.values.iter().map(|x| x + value).collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub_scalar(&self, value: f32) -> Tensor {
        let data = self.values.iter().map(|x| x - value).collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }

    pub fn mul_scalar(&self, value: f32) -> Tensor {
        let data = self.values.iter().map(|x| x * value).collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn item(&self) -> f32 {
        assert_eq!(self.values.len(), 1);
        self.values[0]
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor {
            values: vec![0.0; self.values.len()],
            shape: self.shape.clone(),
        }
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor {
            values: vec![1.0; self.values.len()],
            shape: self.shape.clone(),
        }
    }

    pub fn full_like(&self, value: f32) -> Tensor {
        Tensor {
            values: vec![value; self.values.len()],
            shape: self.shape.clone(),
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let m = self.shape[0];
        let n = self.shape[1];
        let n2 = other.shape[0];
        let p = other.shape[1];
        assert_eq!(n, n2);

        let mut out = vec![0.0; m * p];

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;

                for k in 0..n {
                    sum += self.values[i * n + k] * other.values[k * p + j];
                }
                out[i * p + j] = sum;
            }
        }

        Tensor {
            values: out,
            shape: vec![m, p],
        }
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut out = vec![0.0; self.values.len()];

        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = self.values[i * cols + j];
            }
        }

        Tensor::new(out, vec![cols, rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_broadcasts_trailing_dimensions() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let result = matrix.add(&row);

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.values(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!(result.sum_to_shape(&[3]).values(), &[25.0, 47.0, 69.0]);
    }
}

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

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let data = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let data = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a - b)
            .collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let data = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
    }
    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let data = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a / b)
            .collect();

        Tensor {
            values: data,
            shape: self.shape.clone(),
        }
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

    pub fn matmul(&self, other: Tensor) -> Tensor {
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
                    sum += self.values[i * n + k] + other.values[k * p + j];
                }
                out[i * p + j] = sum;
            }
        }

        Tensor {
            values: out,
            shape: vec![m, p],
        }
    }
}

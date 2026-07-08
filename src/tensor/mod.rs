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
}

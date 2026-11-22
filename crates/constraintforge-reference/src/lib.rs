use std::collections::HashMap;

use constraintforge_ir::{Graph, Inputs, IrError, NodeId, Operation, TensorSpec};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub spec: TensorSpec,
    pub values: Vec<f32>,
}

impl Tensor {
    pub fn new(spec: TensorSpec, values: Vec<f32>) -> Result<Self, ReferenceError> {
        let expected = spec.element_count()?;
        if values.len() != expected {
            return Err(ReferenceError::DataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { spec, values })
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum ReferenceError {
    #[error(transparent)]
    InvalidGraph(#[from] IrError),
    #[error("input for node {0} is missing")]
    MissingInput(usize),
    #[error("tensor requires {expected} values, got {actual}")]
    DataLength { expected: usize, actual: usize },
    #[error("execution did not produce node {0}")]
    MissingValue(usize),
}

pub fn execute(graph: &Graph, inputs: &Inputs) -> Result<HashMap<NodeId, Tensor>, ReferenceError> {
    graph.validate()?;
    let mut values = HashMap::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        let id = graph.node_id(index)?;
        let tensor = match node.operation {
            Operation::Input => Tensor::new(
                node.spec.clone(),
                inputs
                    .get(&id)
                    .cloned()
                    .ok_or(ReferenceError::MissingInput(index))?,
            )?,
            Operation::MatMul { left, right } => matmul(get(&values, left)?, get(&values, right)?)?,
            Operation::BiasAdd { input, bias } => {
                bias_add(get(&values, input)?, get(&values, bias)?)?
            }
            Operation::Relu { input } => relu(get(&values, input)?),
        };
        values.insert(id, tensor);
    }
    Ok(values)
}

pub fn matmul(left: &Tensor, right: &Tensor) -> Result<Tensor, ReferenceError> {
    let [m, k] = dims2(&left.spec)?;
    let [right_k, n] = dims2(&right.spec)?;
    if k != right_k {
        return Err(IrError::MatMulDimensions {
            left: k,
            right: right_k,
        }
        .into());
    }
    let spec = TensorSpec::f32([m, n])?;
    let mut output = vec![0.0; spec.element_count()?];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += left.values[row * k + inner] * right.values[inner * n + col];
            }
            output[row * n + col] = sum;
        }
    }
    Tensor::new(spec, output)
}

pub fn bias_add(input: &Tensor, bias: &Tensor) -> Result<Tensor, ReferenceError> {
    let [m, n] = dims2(&input.spec)?;
    if bias.spec.shape != [n] {
        return Err(IrError::BiasDimensions {
            expected: n,
            actual: bias.spec.shape.clone(),
        }
        .into());
    }
    let mut values = input.values.clone();
    for row in 0..m {
        for col in 0..n {
            values[row * n + col] += bias.values[col];
        }
    }
    Tensor::new(input.spec.clone(), values)
}

pub fn relu(input: &Tensor) -> Tensor {
    Tensor {
        spec: input.spec.clone(),
        values: input.values.iter().map(|&x| x.max(0.0)).collect(),
    }
}

pub fn within_tolerance(actual: f32, expected: f32, absolute: f32, relative: f32) -> bool {
    (actual - expected).abs() <= absolute + relative * expected.abs()
}

fn dims2(spec: &TensorSpec) -> Result<[usize; 2], ReferenceError> {
    spec.shape.as_slice().try_into().map_err(|_| {
        IrError::Rank {
            operation: "matrix operation",
            expected: 2,
            actual: spec.shape.len(),
        }
        .into()
    })
}

fn get(values: &HashMap<NodeId, Tensor>, id: NodeId) -> Result<&Tensor, ReferenceError> {
    values
        .get(&id)
        .ok_or(ReferenceError::MissingValue(id.index()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use constraintforge_ir::{Graph, TensorSpec};

    fn tensor(shape: impl Into<Vec<usize>>, values: &[f32]) -> Tensor {
        Tensor::new(TensorSpec::f32(shape).unwrap(), values.to_vec()).unwrap()
    }

    #[test]
    fn hand_verified_and_rectangular_matmul() {
        let result = matmul(
            &tensor([2, 3], &[1., 2., 3., 4., 5., 6.]),
            &tensor([3, 2], &[7., 8., 9., 10., 11., 12.]),
        )
        .unwrap();
        assert_eq!(result.values, [58., 64., 139., 154.]);
    }

    #[test]
    fn bias_relu_and_invalid_data() {
        let input = tensor([2, 3], &[-2., 0., 2., 4., -4., 1.]);
        let biased = bias_add(&input, &tensor([3], &[1., 2., -3.])).unwrap();
        assert_eq!(relu(&biased).values, [0., 2., 0., 5., 0., 0.]);
        assert!(Tensor::new(TensorSpec::f32([2, 2]).unwrap(), vec![1.]).is_err());
    }

    #[test]
    fn executes_deterministically_end_to_end() {
        let mut graph = Graph::new();
        let a = graph.input(TensorSpec::f32([2, 3]).unwrap());
        let b = graph.input(TensorSpec::f32([3, 2]).unwrap());
        let bias = graph.input(TensorSpec::f32([2]).unwrap());
        let product = graph.matmul(a, b).unwrap();
        let biased = graph.bias_add(product, bias).unwrap();
        let output = graph.relu(biased).unwrap();
        graph.set_outputs(vec![output]).unwrap();
        let inputs = Inputs::from([
            (a, vec![1., 2., 3., 4., 5., 6.]),
            (b, vec![7., 8., 9., 10., 11., 12.]),
            (bias, vec![-60., -60.]),
        ]);
        let first = execute(&graph, &inputs).unwrap();
        let second = execute(&graph, &inputs).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[&output].values, [0., 4., 79., 94.]);
    }

    #[test]
    fn combined_tolerance_is_explicit() {
        assert!(within_tolerance(1000.01, 1000.0, 1e-5, 1e-4));
        assert!(!within_tolerance(1.01, 1.0, 1e-5, 1e-4));
    }
}

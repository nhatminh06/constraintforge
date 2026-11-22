use std::collections::HashMap;

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn f32(shape: impl Into<Vec<usize>>) -> Result<Self, IrError> {
        let spec = Self {
            dtype: DType::F32,
            shape: shape.into(),
        };
        spec.element_count()?;
        Ok(spec)
    }

    pub fn element_count(&self) -> Result<usize, IrError> {
        if self.shape.contains(&0) {
            return Err(IrError::ZeroDimension);
        }
        self.shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or(IrError::ElementCountOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Input,
    MatMul { left: NodeId, right: NodeId },
    BiasAdd { input: NodeId, bias: NodeId },
    Relu { input: NodeId },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub operation: Operation,
    pub spec: TensorSpec,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum IrError {
    #[error("tensor dimensions must be nonzero")]
    ZeroDimension,
    #[error("tensor element count overflows usize")]
    ElementCountOverflow,
    #[error("node {0} does not exist")]
    MissingNode(usize),
    #[error("{operation} requires rank {expected}, got rank {actual}")]
    Rank {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("MatMul inner dimensions differ: left K={left}, right K={right}")]
    MatMulDimensions { left: usize, right: usize },
    #[error("BiasAdd requires bias shape [{expected}], got {actual:?}")]
    BiasDimensions { expected: usize, actual: Vec<usize> },
    #[error("graph output list cannot be empty")]
    NoOutputs,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }
    pub fn node_id(&self, index: usize) -> Result<NodeId, IrError> {
        self.nodes
            .get(index)
            .map(|_| NodeId(index))
            .ok_or(IrError::MissingNode(index))
    }
    pub fn node(&self, id: NodeId) -> Result<&Node, IrError> {
        self.nodes.get(id.0).ok_or(IrError::MissingNode(id.0))
    }
    pub fn input(&mut self, spec: TensorSpec) -> NodeId {
        self.push(Operation::Input, spec)
    }

    pub fn matmul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, IrError> {
        let a = self.node(left)?.spec.clone();
        let b = self.node(right)?.spec.clone();
        require_rank("MatMul left input", &a, 2)?;
        require_rank("MatMul right input", &b, 2)?;
        if a.shape[1] != b.shape[0] {
            return Err(IrError::MatMulDimensions {
                left: a.shape[1],
                right: b.shape[0],
            });
        }
        Ok(self.push(
            Operation::MatMul { left, right },
            TensorSpec::f32([a.shape[0], b.shape[1]])?,
        ))
    }

    pub fn bias_add(&mut self, input: NodeId, bias: NodeId) -> Result<NodeId, IrError> {
        let value = self.node(input)?.spec.clone();
        let bias_spec = self.node(bias)?.spec.clone();
        require_rank("BiasAdd input", &value, 2)?;
        if bias_spec.shape != [value.shape[1]] {
            return Err(IrError::BiasDimensions {
                expected: value.shape[1],
                actual: bias_spec.shape,
            });
        }
        Ok(self.push(Operation::BiasAdd { input, bias }, value))
    }

    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, IrError> {
        let spec = self.node(input)?.spec.clone();
        Ok(self.push(Operation::Relu { input }, spec))
    }

    pub fn set_outputs(&mut self, outputs: impl Into<Vec<NodeId>>) -> Result<(), IrError> {
        let outputs = outputs.into();
        if outputs.is_empty() {
            return Err(IrError::NoOutputs);
        }
        for &output in &outputs {
            self.node(output)?;
        }
        self.outputs = outputs;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), IrError> {
        if self.outputs.is_empty() {
            return Err(IrError::NoOutputs);
        }
        for (index, node) in self.nodes.iter().enumerate() {
            node.spec.element_count()?;
            for dependency in dependencies(&node.operation) {
                if dependency.0 >= index {
                    return Err(IrError::MissingNode(dependency.0));
                }
            }
        }
        for &output in &self.outputs {
            self.node(output)?;
        }
        Ok(())
    }

    fn push(&mut self, operation: Operation, spec: TensorSpec) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { operation, spec });
        id
    }
}

fn require_rank(
    operation: &'static str,
    spec: &TensorSpec,
    expected: usize,
) -> Result<(), IrError> {
    if spec.shape.len() == expected {
        Ok(())
    } else {
        Err(IrError::Rank {
            operation,
            expected,
            actual: spec.shape.len(),
        })
    }
}

fn dependencies(operation: &Operation) -> Vec<NodeId> {
    match *operation {
        Operation::Input => vec![],
        Operation::MatMul { left, right } => vec![left, right],
        Operation::BiasAdd { input, bias } => vec![input, bias],
        Operation::Relu { input } => vec![input],
    }
}

pub type Inputs = HashMap<NodeId, Vec<f32>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_and_infers_a_graph() {
        let mut graph = Graph::new();
        let a = graph.input(TensorSpec::f32([2, 3]).unwrap());
        let b = graph.input(TensorSpec::f32([3, 4]).unwrap());
        let bias = graph.input(TensorSpec::f32([4]).unwrap());
        let product = graph.matmul(a, b).unwrap();
        let biased = graph.bias_add(product, bias).unwrap();
        let output = graph.relu(biased).unwrap();
        graph.set_outputs(vec![output]).unwrap();
        graph.validate().unwrap();
        assert_eq!(graph.node(output).unwrap().spec.shape, [2, 4]);
    }

    #[test]
    fn rejects_invalid_shapes_and_references() {
        assert_eq!(TensorSpec::f32([2, 0]), Err(IrError::ZeroDimension));
        assert_eq!(
            TensorSpec::f32([usize::MAX, 2]),
            Err(IrError::ElementCountOverflow)
        );
        let mut graph = Graph::new();
        let a = graph.input(TensorSpec::f32([2, 3]).unwrap());
        let b = graph.input(TensorSpec::f32([4, 2]).unwrap());
        assert_eq!(
            graph.matmul(a, b),
            Err(IrError::MatMulDimensions { left: 3, right: 4 })
        );
        assert_eq!(graph.node(NodeId(99)), Err(IrError::MissingNode(99)));
    }

    #[test]
    fn rejects_bad_rank_and_bias() {
        let mut graph = Graph::new();
        let vector = graph.input(TensorSpec::f32([3]).unwrap());
        assert!(matches!(
            graph.matmul(vector, vector),
            Err(IrError::Rank { .. })
        ));
        let matrix = graph.input(TensorSpec::f32([2, 3]).unwrap());
        assert_eq!(graph.bias_add(matrix, vector).unwrap(), NodeId(2));
        let bad_bias = graph.input(TensorSpec::f32([2]).unwrap());
        assert!(matches!(
            graph.bias_add(matrix, bad_bias),
            Err(IrError::BiasDimensions { .. })
        ));
    }

    #[test]
    fn validation_rejects_non_topological_dependencies() {
        let graph = Graph {
            nodes: vec![Node {
                operation: Operation::Relu { input: NodeId(0) },
                spec: TensorSpec::f32([1]).unwrap(),
            }],
            outputs: vec![NodeId(0)],
        };
        assert_eq!(graph.validate(), Err(IrError::MissingNode(0)));
    }
}

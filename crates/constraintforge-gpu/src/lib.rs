use std::{
    collections::HashMap,
    sync::mpsc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use constraintforge_ir::{Graph, Inputs, IrError, NodeId, Operation, TensorSpec};
use thiserror::Error;
use wgpu::util::DeviceExt;

const MATRIX_WORKGROUP: u32 = 8;
const ELEMENT_WORKGROUP: u32 = 64;
const TILE: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatMulKernel {
    Naive,
    Tiled,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MatMulKernelVariant {
    Naive,
    Tiled,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelCandidateId {
    pub variant: MatMulKernelVariant,
    pub tile: [u32; 3],
    pub workgroup: [u32; 3],
    pub shader_variant: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateRejection {
    pub id: KernelCandidateId,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatMulKernelConfig {
    variant: MatMulKernelVariant,
    tile: [u32; 3],
    workgroup: [u32; 3],
}

impl MatMulKernelConfig {
    pub const fn naive() -> Self {
        Self {
            variant: MatMulKernelVariant::Naive,
            tile: [1, 1, 1],
            workgroup: [MATRIX_WORKGROUP, MATRIX_WORKGROUP, 1],
        }
    }

    pub const fn tiled() -> Self {
        Self {
            variant: MatMulKernelVariant::Tiled,
            tile: [TILE, TILE, TILE],
            workgroup: [TILE, TILE, 1],
        }
    }

    pub fn tiled_custom(tile: [u32; 3]) -> Result<Self, GpuError> {
        let [bm, bn, bk] = tile;
        if bm == 0 || bn == 0 || bk == 0 {
            return Err(GpuError::InvalidKernelConfig(
                "tile dimensions must be nonzero",
            ));
        }
        Ok(Self {
            variant: MatMulKernelVariant::Tiled,
            tile,
            workgroup: [bn, bm, 1],
        })
    }

    pub const fn candidate_id(self) -> KernelCandidateId {
        KernelCandidateId {
            variant: self.variant,
            tile: self.tile,
            workgroup: self.workgroup,
            shader_variant: match self.variant {
                MatMulKernelVariant::Naive => 0,
                MatMulKernelVariant::Tiled => 1,
            },
        }
    }

    pub const fn variant(self) -> MatMulKernelVariant {
        self.variant
    }

    pub const fn tile(self) -> [u32; 3] {
        self.tile
    }

    pub const fn workgroup(self) -> [u32; 3] {
        self.workgroup
    }

    pub fn shared_memory_bytes(self) -> Result<u32, GpuError> {
        if self.variant == MatMulKernelVariant::Naive {
            return Ok(0);
        }
        let [bm, bn, bk] = self.tile;
        bm.checked_mul(bk)
            .and_then(|a| bk.checked_mul(bn).and_then(|b| a.checked_add(b)))
            .and_then(|elements| elements.checked_mul(size_of::<f32>() as u32))
            .ok_or(GpuError::ResourceOverflow("tiled workgroup memory"))
    }
}

pub fn tiled_matmul_candidates() -> Vec<MatMulKernelConfig> {
    [
        [8, 8, 8],
        [8, 16, 8],
        [16, 8, 8],
        [16, 16, 8],
        [16, 16, 16],
        [16, 16, 32],
        [32, 8, 8],
        [8, 32, 8],
        [32, 8, 16],
        [8, 32, 16],
    ]
    .into_iter()
    .map(|tile| MatMulKernelConfig::tiled_custom(tile).expect("static candidates are nonzero"))
    .collect()
}

pub fn legal_tiled_matmul_candidates(
    shape: [usize; 3],
) -> Result<
    (
        AdapterSummary,
        Vec<MatMulKernelConfig>,
        Vec<CandidateRejection>,
    ),
    GpuError,
> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::AdapterUnavailable)?;
        let limits = adapter.limits();
        let mut legal = Vec::new();
        let mut rejected = Vec::new();
        for config in tiled_matmul_candidates() {
            match validate_kernel(config, shape, &limits) {
                Ok(_) => legal.push(config),
                Err(error) => rejected.push(CandidateRejection {
                    id: config.candidate_id(),
                    reason: error.to_string(),
                }),
            }
        }
        Ok((summarize(&adapter), legal, rejected))
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSelection {
    pub node: NodeId,
    pub shape: [usize; 3],
    pub config: MatMulKernelConfig,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub features: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
}

#[derive(Clone, Debug)]
pub struct PreparedKernelMetadata {
    pub config: MatMulKernelConfig,
    pub dispatch: [u32; 3],
}

#[derive(Debug)]
pub struct PreparedSample {
    pub host_duration_ns: u64,
    pub gpu_compute_pass_duration_ns: Option<f64>,
    pub output: Vec<f32>,
}

struct PreparedKernel {
    metadata: PreparedKernelMetadata,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _output: wgpu::Buffer,
}

pub struct PreparedMatmulBenchmark {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: AdapterSummary,
    preparation_duration: Duration,
    timestamp_period_ns: Option<f32>,
    query_set: Option<wgpu::QuerySet>,
    query_resolve: Option<wgpu::Buffer>,
    query_staging: Option<wgpu::Buffer>,
    output_staging: wgpu::Buffer,
    output_bytes: u64,
    kernels: Vec<PreparedKernel>,
    _left: wgpu::Buffer,
    _right: wgpu::Buffer,
    _params: wgpu::Buffer,
}

impl PreparedMatmulBenchmark {
    pub fn prepare(shape: [usize; 3], left: &[f32], right: &[f32]) -> Result<Self, GpuError> {
        Self::prepare_candidates(
            shape,
            left,
            right,
            &[MatMulKernelConfig::naive(), MatMulKernelConfig::tiled()],
        )
    }

    pub fn prepare_candidates(
        shape: [usize; 3],
        left: &[f32],
        right: &[f32],
        configs: &[MatMulKernelConfig],
    ) -> Result<Self, GpuError> {
        if configs.is_empty() {
            return Err(GpuError::InvalidKernelConfig(
                "at least one benchmark kernel is required",
            ));
        }
        let mut unique = configs.to_vec();
        unique.sort_by_key(|config| {
            let id = config.candidate_id();
            (id.shader_variant, id.tile, id.workgroup)
        });
        unique.dedup_by_key(|config| config.candidate_id());
        pollster::block_on(Self::prepare_async(shape, left, right, &unique))
    }

    async fn prepare_async(
        shape: [usize; 3],
        left_values: &[f32],
        right_values: &[f32],
        configs: &[MatMulKernelConfig],
    ) -> Result<Self, GpuError> {
        let started = Instant::now();
        let [m, k, n] = shape;
        let left_elements = m
            .checked_mul(k)
            .ok_or(GpuError::ResourceOverflow("benchmark left tensor"))?;
        let right_elements = k
            .checked_mul(n)
            .ok_or(GpuError::ResourceOverflow("benchmark right tensor"))?;
        let output_elements = m
            .checked_mul(n)
            .ok_or(GpuError::ResourceOverflow("benchmark output tensor"))?;
        if left_values.len() != left_elements || right_values.len() != right_elements {
            return Err(GpuError::BenchmarkInputLength {
                expected_left: left_elements,
                actual_left: left_values.len(),
                expected_right: right_elements,
                actual_right: right_values.len(),
            });
        }
        let left_bytes = buffer_bytes(left_elements)?;
        let right_bytes = buffer_bytes(right_elements)?;
        let output_bytes = buffer_bytes(output_elements)?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::AdapterUnavailable)?;
        let adapter_limits = adapter.limits();
        let dispatches = configs
            .iter()
            .copied()
            .map(|config| validate_kernel(config, shape, &adapter_limits))
            .collect::<Result<Vec<_>, _>>()?;
        let timestamp_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamp_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let mut required_limits = wgpu::Limits::default();
        let tensor_bytes = left_bytes.max(right_bytes).max(output_bytes);
        required_limits.max_buffer_size = tensor_bytes.max(size_of::<Params>() as u64);
        required_limits.max_storage_buffer_binding_size = tensor_bytes;
        required_limits.max_compute_workgroups_per_dimension =
            dispatches.iter().map(|&(x, y)| x.max(y)).max().unwrap_or(1);
        if !required_limits.check_limits(&adapter_limits) {
            return Err(GpuError::Execution(
                "benchmark resource requirements exceed adapter limits".into(),
            ));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("constraintforge prepared benchmark device"),
                required_features,
                required_limits,
                ..Default::default()
            })
            .await
            .map_err(|error| GpuError::DeviceRequest(error.to_string()))?;

        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let left = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("benchmark left input"),
            contents: bytemuck::cast_slice(left_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let right = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("benchmark right input"),
            contents: bytemuck::cast_slice(right_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = Params {
            x: to_u32("M", m)?,
            y: to_u32("K", k)?,
            z: to_u32("N", n)?,
            _padding: 0,
        };
        validate_operator_parameters(device.limits(), "benchmark MatMul", &params)?;
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("benchmark operator parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut kernels = Vec::with_capacity(configs.len());
        for (config, dispatch) in configs.iter().copied().zip(dispatches) {
            let label = match config.variant {
                MatMulKernelVariant::Naive => "Naive MatMul benchmark".to_owned(),
                MatMulKernelVariant::Tiled => {
                    format!("Tiled MatMul benchmark {:?}", config.tile)
                }
            };
            let shader = match config.variant {
                MatMulKernelVariant::Naive => MATMUL_SHADER.to_owned(),
                MatMulKernelVariant::Tiled => tiled_matmul_shader(config),
            };
            let pipeline = pipeline_source(&device, &label, &shader);
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("benchmark output"),
                size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("benchmark MatMul bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: left.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: right.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            kernels.push(PreparedKernel {
                metadata: PreparedKernelMetadata {
                    config,
                    dispatch: [dispatch.0, dispatch.1, 1],
                },
                pipeline,
                bind_group,
                _output: output,
            });
        }
        let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("benchmark output staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let query_set = timestamp_supported.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("benchmark timestamp queries"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let query_resolve = timestamp_supported.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("benchmark timestamp resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let query_staging = timestamp_supported.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("benchmark timestamp staging"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        if let Some(error) = errors.pop().await {
            return Err(GpuError::Pipeline(error.to_string()));
        }
        let timestamp_period_ns = timestamp_supported.then(|| queue.get_timestamp_period());
        Ok(Self {
            device,
            queue,
            adapter: summarize(&adapter),
            preparation_duration: started.elapsed(),
            timestamp_period_ns,
            query_set,
            query_resolve,
            query_staging,
            output_staging,
            output_bytes,
            kernels,
            _left: left,
            _right: right,
            _params: params_buffer,
        })
    }

    pub fn adapter(&self) -> &AdapterSummary {
        &self.adapter
    }

    pub fn preparation_duration(&self) -> Duration {
        self.preparation_duration
    }

    pub fn timestamp_period_ns(&self) -> Option<f32> {
        self.timestamp_period_ns
    }

    pub fn kernel_metadata(
        &self,
        variant: MatMulKernelVariant,
    ) -> Result<&PreparedKernelMetadata, GpuError> {
        let config = match variant {
            MatMulKernelVariant::Naive => MatMulKernelConfig::naive(),
            MatMulKernelVariant::Tiled => MatMulKernelConfig::tiled(),
        };
        self.kernel_metadata_config(config)
    }

    pub fn kernel_metadata_config(
        &self,
        config: MatMulKernelConfig,
    ) -> Result<&PreparedKernelMetadata, GpuError> {
        self.kernels
            .iter()
            .find(|kernel| kernel.metadata.config == config)
            .map(|kernel| &kernel.metadata)
            .ok_or_else(|| GpuError::Execution("prepared kernel is missing".into()))
    }

    pub fn invoke(&self, variant: MatMulKernelVariant) -> Result<PreparedSample, GpuError> {
        let config = match variant {
            MatMulKernelVariant::Naive => MatMulKernelConfig::naive(),
            MatMulKernelVariant::Tiled => MatMulKernelConfig::tiled(),
        };
        self.invoke_config(config)
    }

    pub fn invoke_config(&self, config: MatMulKernelConfig) -> Result<PreparedSample, GpuError> {
        pollster::block_on(self.invoke_async(config))
    }

    async fn invoke_async(&self, config: MatMulKernelConfig) -> Result<PreparedSample, GpuError> {
        let kernel = self
            .kernels
            .iter()
            .find(|kernel| kernel.metadata.config == config)
            .ok_or_else(|| GpuError::Execution("prepared kernel is missing".into()))?;
        let started = Instant::now();
        let errors = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prepared MatMul invocation"),
            });
        let timestamp_writes =
            self.query_set
                .as_ref()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("measured MatMul compute pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &kernel.bind_group, &[]);
            pass.dispatch_workgroups(
                kernel.metadata.dispatch[0],
                kernel.metadata.dispatch[1],
                kernel.metadata.dispatch[2],
            );
        }
        encoder.copy_buffer_to_buffer(
            &kernel._output,
            0,
            &self.output_staging,
            0,
            self.output_bytes,
        );
        if let (Some(query_set), Some(resolve), Some(staging)) =
            (&self.query_set, &self.query_resolve, &self.query_staging)
        {
            encoder.resolve_query_set(query_set, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, 16);
        }
        let submission = self.queue.submit([encoder.finish()]);
        let output_receiver = map_read(&self.output_staging);
        let query_receiver = self.query_staging.as_ref().map(map_read);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|error| GpuError::Poll(error.to_string()))?;
        if let Some(error) = errors.pop().await {
            return Err(GpuError::Execution(error.to_string()));
        }
        receive_map(output_receiver)?;
        let output_view = self
            .output_staging
            .slice(..)
            .get_mapped_range()
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let output = bytemuck::cast_slice::<u8, f32>(&output_view).to_vec();
        drop(output_view);
        self.output_staging.unmap();

        let gpu_compute_pass_duration_ns = match (&self.query_staging, query_receiver) {
            (Some(staging), Some(receiver)) => {
                receive_map(receiver)?;
                let view = staging
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|error| GpuError::Map(error.to_string()))?;
                let ticks = bytemuck::cast_slice::<u8, u64>(&view);
                let duration = timestamp_duration_ns(
                    ticks[0],
                    ticks[1],
                    self.timestamp_period_ns.ok_or(GpuError::InvalidTimestamp(
                        "timestamp period is unavailable",
                    ))?,
                )?;
                drop(view);
                staging.unmap();
                Some(duration)
            }
            _ => None,
        };
        Ok(PreparedSample {
            host_duration_ns: duration_ns(started.elapsed()),
            gpu_compute_pass_duration_ns,
            output,
        })
    }
}

#[derive(Clone, Debug)]
struct PlannedNode {
    id: NodeId,
    operation: Operation,
    spec: TensorSpec,
    bytes: u64,
    dispatch: Option<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    nodes: Vec<PlannedNode>,
    outputs: Vec<NodeId>,
}

impl ExecutionPlan {
    pub fn prepare(graph: &Graph) -> Result<Self, GpuError> {
        graph.validate()?;
        let mut nodes = Vec::with_capacity(graph.nodes().len());
        for (index, node) in graph.nodes().iter().enumerate() {
            let id = graph.node_id(index)?;
            let bytes = buffer_bytes(node.spec.element_count()?)?;
            let dispatch = match node.operation {
                Operation::Input => None,
                Operation::MatMul { .. } => None,
                Operation::BiasAdd { .. } | Operation::Relu { .. } => {
                    let count = to_u32("element count", node.spec.element_count()?)?;
                    Some((count.div_ceil(ELEMENT_WORKGROUP), 1))
                }
            };
            nodes.push(PlannedNode {
                id,
                operation: node.operation.clone(),
                spec: node.spec.clone(),
                bytes,
                dispatch,
            });
        }
        Ok(Self {
            nodes,
            outputs: graph.outputs().to_vec(),
        })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

#[derive(Debug)]
pub struct GraphExecution {
    pub adapter: AdapterSummary,
    pub outputs: HashMap<NodeId, Vec<f32>>,
    pub matmul_kernels: Vec<KernelSelection>,
}

impl GraphExecution {
    pub fn output(&self, id: NodeId) -> Result<&[f32], GpuError> {
        self.outputs
            .get(&id)
            .map(Vec::as_slice)
            .ok_or(GpuError::MissingOutput(id.index()))
    }
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no compatible GPU adapter is available")]
    AdapterUnavailable,
    #[error("requesting a GPU device failed: {0}")]
    DeviceRequest(String),
    #[error(transparent)]
    InvalidGraph(#[from] IrError),
    #[error("input for node {0} is missing")]
    MissingInput(usize),
    #[error("input was provided for non-input or unknown node {0}")]
    UnexpectedInput(usize),
    #[error("node {node} requires {expected} values, got {actual}")]
    DataLength {
        node: usize,
        expected: usize,
        actual: usize,
    },
    #[error("GPU dimension {name}={value} exceeds u32")]
    DimensionTooLarge { name: &'static str, value: usize },
    #[error("node {node} requires a {requested}-byte buffer, device limit is {limit}")]
    BufferLimit {
        node: usize,
        requested: u64,
        limit: u64,
    },
    #[error("node {node} requires {requested} workgroups, device limit is {limit}")]
    DispatchLimit {
        node: usize,
        requested: u32,
        limit: u32,
    },
    #[error("GPU pipeline creation failed: {0}")]
    Pipeline(String),
    #[error("GPU execution failed: {0}")]
    Execution(String),
    #[error("GPU buffer mapping failed: {0}")]
    Map(String),
    #[error(
        "cannot create {label} for {operation} parameters {parameters:?}: requested {requested} bytes with {usage:?}, device limits are max_buffer_size={max_buffer_size} and max_uniform_buffer_binding_size={max_uniform_binding_size}"
    )]
    BufferCreation {
        operation: &'static str,
        parameters: [u32; 3],
        label: &'static str,
        requested: u64,
        usage: wgpu::BufferUsages,
        max_buffer_size: u64,
        max_uniform_binding_size: u64,
    },
    #[error("waiting for GPU work failed: {0}")]
    Poll(String),
    #[error("GPU mapping callback was not delivered")]
    MapCallback,
    #[error("GPU output for node {0} is missing")]
    MissingOutput(usize),
    #[error(
        "benchmark inputs have wrong lengths: left expected {expected_left}, got {actual_left}; right expected {expected_right}, got {actual_right}"
    )]
    BenchmarkInputLength {
        expected_left: usize,
        actual_left: usize,
        expected_right: usize,
        actual_right: usize,
    },
    #[error("invalid GPU timestamp: {0}")]
    InvalidTimestamp(&'static str),
    #[error("invalid MatMul kernel configuration: {0}")]
    InvalidKernelConfig(&'static str),
    #[error("MatMul kernel resource calculation overflowed: {0}")]
    ResourceOverflow(&'static str),
    #[error("{variant:?} MatMul kernel is unsupported: {reason}")]
    UnsupportedKernel {
        variant: MatMulKernelVariant,
        reason: String,
    },
}

pub fn adapters() -> Vec<AdapterSummary> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .map(|adapter| summarize(&adapter))
            .collect()
    })
}

pub fn select_matmul_kernel(
    request: MatMulKernel,
    node: NodeId,
    shape: [usize; 3],
    limits: &wgpu::Limits,
) -> Result<KernelSelection, GpuError> {
    let naive = MatMulKernelConfig::naive();
    let tiled = MatMulKernelConfig::tiled();
    match request {
        MatMulKernel::Naive => {
            validate_kernel(naive, shape, limits)?;
            Ok(KernelSelection {
                node,
                shape,
                config: naive,
                reason: "naive kernel explicitly requested".into(),
            })
        }
        MatMulKernel::Tiled => {
            validate_kernel(tiled, shape, limits).map_err(|error| GpuError::UnsupportedKernel {
                variant: MatMulKernelVariant::Tiled,
                reason: error.to_string(),
            })?;
            Ok(KernelSelection {
                node,
                shape,
                config: tiled,
                reason: "tiled kernel explicitly requested".into(),
            })
        }
        MatMulKernel::Auto => {
            let [m, k, n] = shape;
            let large_enough = m >= TILE as usize
                && k >= TILE as usize
                && n >= TILE as usize
                && m.checked_mul(n).is_some_and(|elements| elements >= 4096);
            if large_enough && validate_kernel(tiled, shape, limits).is_ok() {
                Ok(KernelSelection {
                    node,
                    shape,
                    config: tiled,
                    reason: "AUTO selected tiled: all dimensions are at least 16 and M*N is at least 4096".into(),
                })
            } else {
                validate_kernel(naive, shape, limits)?;
                let reason = if large_enough {
                    "AUTO selected naive: tiled resources exceed device limits"
                } else {
                    "AUTO selected naive: shape is below the conservative tiled threshold"
                };
                Ok(KernelSelection {
                    node,
                    shape,
                    config: naive,
                    reason: reason.into(),
                })
            }
        }
    }
}

fn validate_kernel(
    config: MatMulKernelConfig,
    shape: [usize; 3],
    limits: &wgpu::Limits,
) -> Result<(u32, u32), GpuError> {
    let [bm, bn, bk] = config.tile;
    let [x, y, z] = config.workgroup;
    if [bm, bn, bk, x, y, z].contains(&0) {
        return Err(GpuError::InvalidKernelConfig(
            "tile and workgroup dimensions must be nonzero",
        ));
    }
    let invocations = x
        .checked_mul(y)
        .and_then(|xy| xy.checked_mul(z))
        .ok_or(GpuError::ResourceOverflow("workgroup invocations"))?;
    if x > limits.max_compute_workgroup_size_x
        || y > limits.max_compute_workgroup_size_y
        || z > limits.max_compute_workgroup_size_z
        || invocations > limits.max_compute_invocations_per_workgroup
    {
        return Err(GpuError::UnsupportedKernel {
            variant: config.variant,
            reason: format!(
                "workgroup {:?} ({invocations} invocations) exceeds device limits [{}, {}, {}] / {}",
                config.workgroup,
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
                limits.max_compute_invocations_per_workgroup
            ),
        });
    }
    let shared = config.shared_memory_bytes()?;
    if shared > limits.max_compute_workgroup_storage_size {
        return Err(GpuError::UnsupportedKernel {
            variant: config.variant,
            reason: format!(
                "{shared} bytes of workgroup memory exceeds {}",
                limits.max_compute_workgroup_storage_size
            ),
        });
    }
    let m = to_u32("M", shape[0])?;
    let n = to_u32("N", shape[2])?;
    let dispatch = match config.variant {
        MatMulKernelVariant::Naive => (
            n.div_ceil(config.workgroup[0]),
            m.div_ceil(config.workgroup[1]),
        ),
        MatMulKernelVariant::Tiled => (n.div_ceil(bn), m.div_ceil(bm)),
    };
    if dispatch.0 > limits.max_compute_workgroups_per_dimension
        || dispatch.1 > limits.max_compute_workgroups_per_dimension
    {
        return Err(GpuError::UnsupportedKernel {
            variant: config.variant,
            reason: format!(
                "dispatch {:?} exceeds {} workgroups per dimension",
                dispatch, limits.max_compute_workgroups_per_dimension
            ),
        });
    }
    Ok(dispatch)
}

pub fn execute_graph(graph: &Graph, inputs: &Inputs) -> Result<GraphExecution, GpuError> {
    execute_graph_with_kernel(graph, inputs, MatMulKernel::Naive)
}

pub fn execute_graph_with_kernel(
    graph: &Graph,
    inputs: &Inputs,
    kernel: MatMulKernel,
) -> Result<GraphExecution, GpuError> {
    let plan = ExecutionPlan::prepare(graph)?;
    pollster::block_on(execute_plan(&plan, inputs, kernel))
}

pub fn matmul(
    left_shape: [usize; 2],
    left: &[f32],
    right_shape: [usize; 2],
    right: &[f32],
) -> Result<Vec<f32>, GpuError> {
    let mut graph = Graph::new();
    let left_id = graph.input(TensorSpec::f32(left_shape)?);
    let right_id = graph.input(TensorSpec::f32(right_shape)?);
    let output = graph.matmul(left_id, right_id)?;
    graph.set_outputs(vec![output])?;
    let execution = execute_graph(
        &graph,
        &Inputs::from([(left_id, left.to_vec()), (right_id, right.to_vec())]),
    )?;
    execution
        .outputs
        .get(&output)
        .cloned()
        .ok_or(GpuError::MissingOutput(output.index()))
}

async fn execute_plan(
    plan: &ExecutionPlan,
    inputs: &Inputs,
    kernel: MatMulKernel,
) -> Result<GraphExecution, GpuError> {
    validate_inputs(plan, inputs)?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        })
        .await
        .map_err(|_| GpuError::AdapterUnavailable)?;
    let adapter_limits = adapter.limits();
    validate_limits(plan, &adapter_limits)?;
    let matmul_kernels = plan
        .nodes
        .iter()
        .filter_map(|node| match node.operation {
            Operation::MatMul { left, .. } => Some((node, left)),
            _ => None,
        })
        .map(|(node, left)| {
            let [m, n] = matrix_shape(&node.spec, "MatMul output")?;
            let [_left_m, k] = matrix_shape(&plan.nodes[left.index()].spec, "MatMul left input")?;
            select_matmul_kernel(kernel, node.id, [m, k, n], &adapter_limits)
        })
        .collect::<Result<Vec<_>, GpuError>>()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("constraintforge graph device"),
            required_features: wgpu::Features::empty(),
            required_limits: required_limits(plan, &matmul_kernels),
            ..Default::default()
        })
        .await
        .map_err(|error| GpuError::DeviceRequest(error.to_string()))?;

    let pipeline_errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let matmul_pipeline = pipeline(&device, "MatMul", MATMUL_SHADER);
    let tiled_pipeline = matmul_kernels
        .iter()
        .any(|selection| selection.config.variant == MatMulKernelVariant::Tiled)
        .then(|| pipeline(&device, "Tiled MatMul", TILED_MATMUL_SHADER));
    let bias_pipeline = pipeline(&device, "BiasAdd", BIAS_ADD_SHADER);
    let relu_pipeline = pipeline(&device, "ReLU", RELU_SHADER);
    if let Some(error) = pipeline_errors.pop().await {
        return Err(GpuError::Pipeline(error.to_string()));
    }

    let mut buffers = Vec::with_capacity(plan.nodes.len());
    for node in &plan.nodes {
        let buffer = match node.operation {
            Operation::Input => {
                let values = inputs
                    .get(&node.id)
                    .ok_or(GpuError::MissingInput(node.id.index()))?;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("graph input"),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE
                        | if plan.outputs.contains(&node.id) {
                            wgpu::BufferUsages::COPY_SRC
                        } else {
                            wgpu::BufferUsages::empty()
                        },
                })
            }
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("graph tensor"),
                size: node.bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | if plan.outputs.contains(&node.id) {
                        wgpu::BufferUsages::COPY_SRC
                    } else {
                        wgpu::BufferUsages::empty()
                    },
                mapped_at_creation: false,
            }),
        };
        buffers.push(buffer);
    }

    let execution_errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("graph execution"),
    });
    for node in &plan.nodes {
        match node.operation {
            Operation::Input => {}
            Operation::MatMul { left, right } => {
                let [m, n] = matrix_shape(&node.spec, "MatMul output")?;
                let [_left_m, k] =
                    matrix_shape(&plan.nodes[left.index()].spec, "MatMul left input")?;
                let params = Params {
                    x: to_u32("M", m)?,
                    y: to_u32("K", k)?,
                    z: to_u32("N", n)?,
                    _padding: 0,
                };
                let selection = matmul_kernels
                    .iter()
                    .find(|selection| selection.node == node.id)
                    .ok_or_else(|| {
                        GpuError::Execution("MatMul kernel selection is missing".into())
                    })?;
                let dispatch_size =
                    validate_kernel(selection.config, selection.shape, &adapter_limits)?;
                let selected_pipeline = match selection.config.variant {
                    MatMulKernelVariant::Naive => &matmul_pipeline,
                    MatMulKernelVariant::Tiled => tiled_pipeline.as_ref().ok_or_else(|| {
                        GpuError::Execution("tiled MatMul pipeline is missing".into())
                    })?,
                };
                dispatch(
                    &device,
                    &mut encoder,
                    "MatMul",
                    selected_pipeline,
                    &[
                        &buffers[left.index()],
                        &buffers[right.index()],
                        &buffers[node.id.index()],
                    ],
                    &params,
                    Some(dispatch_size),
                )?;
            }
            Operation::BiasAdd { input, bias } => {
                let [_m, n] = matrix_shape(&node.spec, "BiasAdd output")?;
                let params = Params {
                    x: to_u32("element count", node.spec.element_count()?)?,
                    y: to_u32("BiasAdd N", n)?,
                    z: 0,
                    _padding: 0,
                };
                dispatch(
                    &device,
                    &mut encoder,
                    "BiasAdd",
                    &bias_pipeline,
                    &[
                        &buffers[input.index()],
                        &buffers[bias.index()],
                        &buffers[node.id.index()],
                    ],
                    &params,
                    node.dispatch,
                )?;
            }
            Operation::Relu { input } => {
                let params = Params {
                    x: to_u32("element count", node.spec.element_count()?)?,
                    y: 0,
                    z: 0,
                    _padding: 0,
                };
                dispatch(
                    &device,
                    &mut encoder,
                    "ReLU",
                    &relu_pipeline,
                    &[&buffers[input.index()], &buffers[node.id.index()]],
                    &params,
                    node.dispatch,
                )?;
            }
        }
    }

    let mut readbacks = Vec::with_capacity(plan.outputs.len());
    for output in &plan.outputs {
        let node = &plan.nodes[output.index()];
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("graph output readback"),
            size: node.bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffers[output.index()], 0, &readback, 0, node.bytes);
        readbacks.push(readback);
    }
    let submission = queue.submit([encoder.finish()]);
    let mut receivers = Vec::with_capacity(readbacks.len());
    for readback in &readbacks {
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        receivers.push(receiver);
    }
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|error| GpuError::Poll(error.to_string()))?;
    if let Some(error) = execution_errors.pop().await {
        return Err(GpuError::Execution(error.to_string()));
    }

    let mut outputs = HashMap::with_capacity(plan.outputs.len());
    for ((id, readback), receiver) in plan.outputs.iter().zip(&readbacks).zip(receivers) {
        receiver
            .recv()
            .map_err(|_| GpuError::MapCallback)?
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let slice = readback.slice(..);
        let view = slice
            .get_mapped_range()
            .map_err(|error| GpuError::Map(error.to_string()))?;
        outputs.insert(*id, bytemuck::cast_slice::<u8, f32>(&view).to_vec());
        drop(view);
        readback.unmap();
    }
    Ok(GraphExecution {
        adapter: summarize(&adapter),
        outputs,
        matmul_kernels,
    })
}

fn validate_inputs(plan: &ExecutionPlan, inputs: &Inputs) -> Result<(), GpuError> {
    for (&id, values) in inputs {
        let Some(node) = plan.nodes.get(id.index()) else {
            return Err(GpuError::UnexpectedInput(id.index()));
        };
        if node.operation != Operation::Input {
            return Err(GpuError::UnexpectedInput(id.index()));
        }
        let expected = node.spec.element_count()?;
        if values.len() != expected {
            return Err(GpuError::DataLength {
                node: id.index(),
                expected,
                actual: values.len(),
            });
        }
    }
    for node in &plan.nodes {
        if node.operation == Operation::Input && !inputs.contains_key(&node.id) {
            return Err(GpuError::MissingInput(node.id.index()));
        }
    }
    Ok(())
}

fn validate_limits(plan: &ExecutionPlan, limits: &wgpu::Limits) -> Result<(), GpuError> {
    for node in &plan.nodes {
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if node.bytes > limit {
            return Err(GpuError::BufferLimit {
                node: node.id.index(),
                requested: node.bytes,
                limit,
            });
        }
        if let Some((x, y)) = node.dispatch {
            for requested in [x, y] {
                if requested > limits.max_compute_workgroups_per_dimension {
                    return Err(GpuError::DispatchLimit {
                        node: node.id.index(),
                        requested,
                        limit: limits.max_compute_workgroups_per_dimension,
                    });
                }
            }
        }
    }
    Ok(())
}

fn required_limits(plan: &ExecutionPlan, selections: &[KernelSelection]) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    let tensor_bytes = plan.nodes.iter().map(|node| node.bytes).max().unwrap_or(4);
    let bytes = tensor_bytes.max(size_of::<Params>() as u64);
    let element_dispatch = plan
        .nodes
        .iter()
        .filter_map(|node| node.dispatch)
        .map(|(x, y)| x.max(y))
        .max()
        .unwrap_or(1);
    let matmul_dispatch = selections
        .iter()
        .map(|selection| {
            let [m, _, n] = selection.shape;
            let [bm, bn, _] = selection.config.tile;
            let divisor_x = if selection.config.variant == MatMulKernelVariant::Tiled {
                bn
            } else {
                selection.config.workgroup[0]
            };
            let divisor_y = if selection.config.variant == MatMulKernelVariant::Tiled {
                bm
            } else {
                selection.config.workgroup[1]
            };
            (n as u32)
                .div_ceil(divisor_x)
                .max((m as u32).div_ceil(divisor_y))
        })
        .max()
        .unwrap_or(1);
    limits.max_buffer_size = bytes;
    limits.max_storage_buffer_binding_size = tensor_bytes;
    limits.max_compute_workgroups_per_dimension = element_dispatch.max(matmul_dispatch);
    limits
}

fn pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn pipeline_source(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn tiled_matmul_shader(config: MatMulKernelConfig) -> String {
    let [bm, bn, bk] = config.tile;
    let left_elements = bm * bk;
    let right_elements = bk * bn;
    let invocations = bm * bn;
    format!(
        r#"
struct Params {{ m: u32, k: u32, n: u32, padding: u32 }}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Params;
var<workgroup> left_tile: array<f32, {left_elements}>;
var<workgroup> right_tile: array<f32, {right_elements}>;

@compute @workgroup_size({bn}, {bm}, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {{
    let row = group.y * {bm}u + local.y;
    let col = group.x * {bn}u + local.x;
    let local_index = local.y * {bn}u + local.x;
    let tile_count = (dims.k + {bk_minus_one}u) / {bk}u;
    var sum = 0.0;

    for (var tile = 0u; tile < tile_count; tile++) {{
        for (var index = local_index; index < {left_elements}u; index += {invocations}u) {{
            let tile_row = index / {bk}u;
            let tile_col = index % {bk}u;
            let global_row = group.y * {bm}u + tile_row;
            let global_col = tile * {bk}u + tile_col;
            var value = 0.0;
            if (global_row < dims.m && global_col < dims.k) {{
                value = left[global_row * dims.k + global_col];
            }}
            left_tile[index] = value;
        }}
        for (var index = local_index; index < {right_elements}u; index += {invocations}u) {{
            let tile_row = index / {bn}u;
            let tile_col = index % {bn}u;
            let global_row = tile * {bk}u + tile_row;
            let global_col = group.x * {bn}u + tile_col;
            var value = 0.0;
            if (global_row < dims.k && global_col < dims.n) {{
                value = right[global_row * dims.n + global_col];
            }}
            right_tile[index] = value;
        }}
        workgroupBarrier();
        for (var inner = 0u; inner < {bk}u; inner++) {{
            sum += left_tile[local.y * {bk}u + inner] * right_tile[inner * {bn}u + local.x];
        }}
        workgroupBarrier();
    }}
    if (row < dims.m && col < dims.n) {{
        output[row * dims.n + col] = sum;
    }}
}}
"#,
        bk_minus_one = bk - 1,
    )
}

fn dispatch(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    operation: &'static str,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
    params: &Params,
    workgroups: Option<(u32, u32)>,
) -> Result<(), GpuError> {
    validate_operator_parameters(device.limits(), operation, params)?;
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("operator parameters"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: params_buffer.as_entire_binding(),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("operator bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let (x, y) = workgroups
        .ok_or_else(|| GpuError::Execution("operator has no dispatch dimensions".into()))?;
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("graph operator"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(x, y, 1);
    Ok(())
}

fn validate_operator_parameters(
    limits: wgpu::Limits,
    operation: &'static str,
    params: &Params,
) -> Result<(), GpuError> {
    let requested = size_of::<Params>() as u64;
    if requested > limits.max_buffer_size || requested > limits.max_uniform_buffer_binding_size {
        return Err(GpuError::BufferCreation {
            operation,
            parameters: [params.x, params.y, params.z],
            label: "operator parameters",
            requested,
            usage: wgpu::BufferUsages::UNIFORM,
            max_buffer_size: limits.max_buffer_size,
            max_uniform_binding_size: limits.max_uniform_buffer_binding_size,
        });
    }
    Ok(())
}

fn map_read(buffer: &wgpu::Buffer) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    receiver
}

fn receive_map(
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
) -> Result<(), GpuError> {
    receiver
        .recv()
        .map_err(|_| GpuError::MapCallback)?
        .map_err(|error| GpuError::Map(error.to_string()))
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub fn timestamp_duration_ns(start: u64, end: u64, period_ns: f32) -> Result<f64, GpuError> {
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return Err(GpuError::InvalidTimestamp(
            "timestamp period must be positive and finite",
        ));
    }
    let ticks = end
        .checked_sub(start)
        .ok_or(GpuError::InvalidTimestamp("timestamp interval is reversed"))?;
    let duration = ticks as f64 * f64::from(period_ns);
    if !duration.is_finite() {
        return Err(GpuError::InvalidTimestamp(
            "timestamp duration is not finite",
        ));
    }
    Ok(duration)
}

fn summarize(adapter: &wgpu::Adapter) -> AdapterSummary {
    let info = adapter.get_info();
    let limits = adapter.limits();
    AdapterSummary {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        features: format!("{:?}", adapter.features()),
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    }
}

fn matrix_shape(spec: &TensorSpec, operation: &'static str) -> Result<[usize; 2], GpuError> {
    spec.shape.as_slice().try_into().map_err(|_| {
        IrError::Rank {
            operation,
            expected: 2,
            actual: spec.shape.len(),
        }
        .into()
    })
}

fn to_u32(name: &'static str, value: usize) -> Result<u32, GpuError> {
    value
        .try_into()
        .map_err(|_| GpuError::DimensionTooLarge { name, value })
}

fn buffer_bytes(elements: usize) -> Result<u64, GpuError> {
    elements
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or(IrError::ElementCountOverflow.into())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    x: u32,
    y: u32,
    z: u32,
    _padding: u32,
}

const MATMUL_SHADER: &str = r#"
struct Params { m: u32, k: u32, n: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x; let row = id.y;
    if (row >= dims.m || col >= dims.n) { return; }
    var sum = 0.0;
    for (var inner = 0u; inner < dims.k; inner++) {
        sum += left[row * dims.k + inner] * right[inner * dims.n + col];
    }
    output[row * dims.n + col] = sum;
}
"#;

const TILED_MATMUL_SHADER: &str = r#"
struct Params { m: u32, k: u32, n: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Params;
var<workgroup> left_tile: array<f32, 256>;
var<workgroup> right_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = group.y * 16u + local.y;
    let col = group.x * 16u + local.x;
    let local_index = local.y * 16u + local.x;
    let tile_count = (dims.k + 15u) / 16u;
    var sum = 0.0;

    for (var tile = 0u; tile < tile_count; tile++) {
        let left_col = tile * 16u + local.x;
        let right_row = tile * 16u + local.y;
        var left_value = 0.0;
        var right_value = 0.0;
        if (row < dims.m && left_col < dims.k) {
            left_value = left[row * dims.k + left_col];
        }
        if (right_row < dims.k && col < dims.n) {
            right_value = right[right_row * dims.n + col];
        }
        left_tile[local_index] = left_value;
        right_tile[local_index] = right_value;
        workgroupBarrier();

        for (var inner = 0u; inner < 16u; inner++) {
            sum += left_tile[local.y * 16u + inner] * right_tile[inner * 16u + local.x];
        }
        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        output[row * dims.n + col] = sum;
    }
}
"#;

const BIAS_ADD_SHADER: &str = r#"
struct Params { count: u32, n: u32, unused: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < dims.count) { output[index] = input[index] + bias[index % dims.n]; }
}
"#;

const RELU_SHADER: &str = r#"
struct Params { count: u32, unused1: u32, unused2: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < dims.count) { output[index] = max(input[index], 0.0); }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use constraintforge_reference::{execute, within_tolerance};

    fn mlp(batch: usize) -> (Graph, Inputs, NodeId) {
        let mut graph = Graph::new();
        let x = graph.input(TensorSpec::f32([batch, 3]).unwrap());
        let w1 = graph.input(TensorSpec::f32([3, 4]).unwrap());
        let b1 = graph.input(TensorSpec::f32([4]).unwrap());
        let w2 = graph.input(TensorSpec::f32([4, 2]).unwrap());
        let b2 = graph.input(TensorSpec::f32([2]).unwrap());
        let hidden = graph.matmul(x, w1).unwrap();
        let hidden = graph.bias_add(hidden, b1).unwrap();
        let hidden = graph.relu(hidden).unwrap();
        let output = graph.matmul(hidden, w2).unwrap();
        let output = graph.bias_add(output, b2).unwrap();
        graph.set_outputs(vec![hidden, output]).unwrap();
        let x_values = (0..batch * 3).map(|i| i as f32 / 3.0 - 1.0).collect();
        let inputs = Inputs::from([
            (x, x_values),
            (
                w1,
                vec![0.5, -1., 0., 2., -0.5, 1., 1.5, 0., 1., 0.25, -2., 0.5],
            ),
            (b1, vec![0., -0.5, 1., 0.25]),
            (w2, vec![1., -1., 0.5, 2., -0.5, 1., 2., 0.25]),
            (b2, vec![-0.25, 0.5]),
        ]);
        (graph, inputs, output)
    }

    #[test]
    fn plan_tracks_nodes_outputs_and_resources() {
        let (graph, inputs, _) = mlp(3);
        let plan = ExecutionPlan::prepare(&graph).unwrap();
        assert_eq!(plan.node_count(), 10);
        assert_eq!(plan.output_count(), 2);
        validate_inputs(&plan, &inputs).unwrap();
        let tiny = wgpu::Limits {
            max_storage_buffer_binding_size: 4,
            ..Default::default()
        };
        assert!(matches!(
            validate_limits(&plan, &tiny),
            Err(GpuError::BufferLimit { .. })
        ));
        assert_eq!(
            plan.nodes
                .iter()
                .filter(|node| node.operation == Operation::Input)
                .count(),
            inputs.len()
        );
    }

    #[test]
    fn device_limits_cover_operator_parameter_buffers() {
        let mut graph = Graph::new();
        let left = graph.input(TensorSpec::f32([1, 1]).unwrap());
        let right = graph.input(TensorSpec::f32([1, 1]).unwrap());
        let output = graph.matmul(left, right).unwrap();
        graph.set_outputs(vec![output]).unwrap();
        let plan = ExecutionPlan::prepare(&graph).unwrap();
        let limits = required_limits(&plan, &[]);
        assert_eq!(limits.max_buffer_size, size_of::<Params>() as u64);
        assert_eq!(limits.max_storage_buffer_binding_size, 4);

        let too_small = wgpu::Limits {
            max_buffer_size: 4,
            ..wgpu::Limits::default()
        };
        assert!(matches!(
            validate_operator_parameters(
                too_small,
                "MatMul",
                &Params {
                    x: 1,
                    y: 1,
                    z: 1,
                    _padding: 0,
                }
            ),
            Err(GpuError::BufferCreation {
                requested: 16,
                parameters: [1, 1, 1],
                ..
            })
        ));
    }

    #[test]
    fn rejects_missing_extra_and_wrong_length_inputs() {
        let (graph, mut inputs, _) = mlp(1);
        let plan = ExecutionPlan::prepare(&graph).unwrap();
        let first = graph.node_id(0).unwrap();
        inputs.remove(&first);
        assert!(matches!(
            validate_inputs(&plan, &inputs),
            Err(GpuError::MissingInput(0))
        ));
        inputs.insert(first, vec![1.]);
        assert!(matches!(
            validate_inputs(&plan, &inputs),
            Err(GpuError::DataLength { .. })
        ));
        inputs.insert(first, vec![-1., -2. / 3., -1. / 3.]);
        let output = *graph.outputs().last().unwrap();
        inputs.insert(output, vec![1., 2.]);
        assert!(matches!(
            validate_inputs(&plan, &inputs),
            Err(GpuError::UnexpectedInput(_))
        ));
    }

    #[test]
    fn validates_kernel_resources_and_auto_selection() {
        let limits = wgpu::Limits::default();
        let tiled = MatMulKernelConfig::tiled();
        let node = mlp(1).0.node_id(0).unwrap();
        assert_eq!(tiled.shared_memory_bytes().unwrap(), 2048);
        assert_eq!(
            validate_kernel(tiled, [31, 33, 29], &limits).unwrap(),
            (2, 2)
        );

        let invalid = MatMulKernelConfig {
            tile: [0, 16, 16],
            ..tiled
        };
        assert!(matches!(
            validate_kernel(invalid, [32, 32, 32], &limits),
            Err(GpuError::InvalidKernelConfig(_))
        ));
        let overflowing = MatMulKernelConfig {
            tile: [u32::MAX, u32::MAX, u32::MAX],
            ..tiled
        };
        assert!(matches!(
            overflowing.shared_memory_bytes(),
            Err(GpuError::ResourceOverflow(_))
        ));

        let invocation_limited = wgpu::Limits {
            max_compute_invocations_per_workgroup: 128,
            ..limits.clone()
        };
        assert!(matches!(
            validate_kernel(tiled, [32, 32, 32], &invocation_limited),
            Err(GpuError::UnsupportedKernel { .. })
        ));
        assert!(matches!(
            select_matmul_kernel(MatMulKernel::Tiled, node, [32, 32, 32], &invocation_limited),
            Err(GpuError::UnsupportedKernel { .. })
        ));

        let memory_limited = wgpu::Limits {
            max_compute_workgroup_storage_size: 1024,
            ..limits.clone()
        };
        assert!(matches!(
            validate_kernel(tiled, [32, 32, 32], &memory_limited),
            Err(GpuError::UnsupportedKernel { .. })
        ));

        let dispatch_limited = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..limits.clone()
        };
        assert!(matches!(
            validate_kernel(tiled, [17, 16, 16], &dispatch_limited),
            Err(GpuError::UnsupportedKernel { .. })
        ));

        let small = select_matmul_kernel(MatMulKernel::Auto, node, [15, 17, 13], &limits).unwrap();
        let large = select_matmul_kernel(MatMulKernel::Auto, node, [64, 64, 64], &limits).unwrap();
        assert_eq!(small.config.variant(), MatMulKernelVariant::Naive);
        assert_eq!(large.config.variant(), MatMulKernelVariant::Tiled);
        let constrained =
            select_matmul_kernel(MatMulKernel::Auto, node, [64, 64, 64], &memory_limited).unwrap();
        assert_eq!(constrained.config.variant(), MatMulKernelVariant::Naive);
        assert_eq!(large.config.tile(), [16, 16, 16]);
        assert_eq!(large.config.workgroup(), [16, 16, 1]);
        assert_eq!(
            large,
            select_matmul_kernel(MatMulKernel::Auto, node, [64, 64, 64], &limits).unwrap()
        );
        assert!(large.reason.contains("AUTO selected tiled"));
    }

    #[test]
    fn configurable_candidates_are_unique_and_change_generated_work() {
        let candidates = tiled_matmul_candidates();
        let identities = candidates
            .iter()
            .map(|config| config.candidate_id())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(candidates.len(), 10);
        assert_eq!(identities.len(), candidates.len());
        assert!(MatMulKernelConfig::tiled_custom([0, 8, 8]).is_err());
        for config in &candidates {
            let [bm, bn, bk] = config.tile();
            assert_eq!(config.workgroup(), [bn, bm, 1]);
            assert_eq!(
                config.shared_memory_bytes().unwrap(),
                4 * (bm * bk + bk * bn)
            );
        }
        let eight = tiled_matmul_shader(MatMulKernelConfig::tiled_custom([8, 8, 8]).unwrap());
        let deep = tiled_matmul_shader(MatMulKernelConfig::tiled_custom([16, 16, 32]).unwrap());
        assert!(eight.contains("@workgroup_size(8, 8, 1)"));
        assert!(deep.contains("@workgroup_size(16, 16, 1)"));
        assert!(deep.contains("inner < 32u"));
        assert_ne!(eight, deep);
    }

    #[test]
    #[ignore = "requires a real GPU adapter; run explicitly with --ignored"]
    fn real_gpu_configurable_tiled_candidates_match_cpu() {
        for shape in [[2, 3, 4], [15, 17, 13], [31, 33, 29]] {
            let left = (0..shape[0] * shape[1])
                .map(|index| (index % 11) as f32 / 5.0 - 1.0)
                .collect::<Vec<_>>();
            let right = (0..shape[1] * shape[2])
                .map(|index| (index % 7) as f32 / 3.0 - 1.0)
                .collect::<Vec<_>>();
            let expected = constraintforge_reference::matmul(
                &constraintforge_reference::Tensor::new(
                    TensorSpec::f32([shape[0], shape[1]]).unwrap(),
                    left.clone(),
                )
                .unwrap(),
                &constraintforge_reference::Tensor::new(
                    TensorSpec::f32([shape[1], shape[2]]).unwrap(),
                    right.clone(),
                )
                .unwrap(),
            )
            .unwrap();
            let candidates = tiled_matmul_candidates();
            let prepared =
                PreparedMatmulBenchmark::prepare_candidates(shape, &left, &right, &candidates)
                    .unwrap();
            for candidate in candidates {
                let sample = prepared.invoke_config(candidate).unwrap();
                assert!(
                    sample
                        .output
                        .iter()
                        .zip(&expected.values)
                        .all(|(&actual, &reference)| actual.is_finite()
                            && within_tolerance(actual, reference, 1e-5, 1e-4))
                );
            }
        }
    }

    #[test]
    fn validates_timestamp_conversion() {
        assert_eq!(timestamp_duration_ns(10, 14, 2.5).unwrap(), 10.0);
        assert!(matches!(
            timestamp_duration_ns(14, 10, 1.0),
            Err(GpuError::InvalidTimestamp(_))
        ));
        for period in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                timestamp_duration_ns(10, 14, period),
                Err(GpuError::InvalidTimestamp(_))
            ));
        }
    }

    #[test]
    #[ignore = "requires a real GPU adapter; run explicitly with --ignored"]
    fn real_gpu_prepared_benchmark_timestamps_and_correctness() {
        let shape = [31, 33, 29];
        let left = (0..shape[0] * shape[1])
            .map(|index| (index % 11) as f32 / 5.0 - 1.0)
            .collect::<Vec<_>>();
        let right = (0..shape[1] * shape[2])
            .map(|index| (index % 7) as f32 / 3.0 - 1.0)
            .collect::<Vec<_>>();
        let expected = constraintforge_reference::matmul(
            &constraintforge_reference::Tensor::new(
                TensorSpec::f32([shape[0], shape[1]]).unwrap(),
                left.clone(),
            )
            .unwrap(),
            &constraintforge_reference::Tensor::new(
                TensorSpec::f32([shape[1], shape[2]]).unwrap(),
                right.clone(),
            )
            .unwrap(),
        )
        .unwrap();
        let prepared = PreparedMatmulBenchmark::prepare(shape, &left, &right).unwrap();
        for variant in [MatMulKernelVariant::Naive, MatMulKernelVariant::Tiled] {
            prepared.invoke(variant).unwrap();
            for _ in 0..3 {
                let sample = prepared.invoke(variant).unwrap();
                assert_eq!(sample.output.len(), expected.values.len());
                assert!(
                    sample
                        .output
                        .iter()
                        .zip(&expected.values)
                        .all(|(&actual, &reference)| actual.is_finite()
                            && within_tolerance(actual, reference, 1e-5, 1e-4))
                );
                match sample.gpu_compute_pass_duration_ns {
                    Some(duration) => assert!(duration.is_finite() && duration >= 0.0),
                    None => assert!(prepared.timestamp_period_ns().is_none()),
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a real GPU adapter; run explicitly with --ignored"]
    fn real_gpu_executes_operators_outputs_and_mlp() {
        for kernel in [MatMulKernel::Naive, MatMulKernel::Tiled, MatMulKernel::Auto] {
            for batch in [1, 3, 9] {
                let (graph, inputs, output) = mlp(batch);
                let expected = execute(&graph, &inputs).unwrap();
                let first = execute_graph_with_kernel(&graph, &inputs, kernel).unwrap();
                let second = execute_graph_with_kernel(&graph, &inputs, kernel).unwrap();
                assert_eq!(first.outputs, second.outputs);
                for &id in graph.outputs() {
                    assert!(
                        first.outputs[&id]
                            .iter()
                            .zip(&expected[&id].values)
                            .all(|(&actual, &reference)| actual.is_finite()
                                && within_tolerance(actual, reference, 1e-5, 1e-4))
                    );
                }
                assert_eq!(first.outputs[&output].len(), batch * 2);
            }
        }

        for [m, k, n] in [
            [1, 1, 1],
            [2, 2, 2],
            [15, 15, 15],
            [15, 16, 13],
            [15, 17, 13],
            [31, 33, 29],
        ] {
            let mut graph = Graph::new();
            let left = graph.input(TensorSpec::f32([m, k]).unwrap());
            let right = graph.input(TensorSpec::f32([k, n]).unwrap());
            let output = graph.matmul(left, right).unwrap();
            graph.set_outputs(vec![output]).unwrap();
            let inputs = Inputs::from([
                (left, (0..m * k).map(|i| (i % 11) as f32 - 5.0).collect()),
                (
                    right,
                    (0..k * n).map(|i| (i % 7) as f32 / 3.0 - 1.0).collect(),
                ),
            ]);
            let expected = execute(&graph, &inputs).unwrap();
            let naive = execute_graph_with_kernel(&graph, &inputs, MatMulKernel::Naive).unwrap();
            let tiled = execute_graph_with_kernel(&graph, &inputs, MatMulKernel::Tiled).unwrap();
            for actual in [naive.output(output).unwrap(), tiled.output(output).unwrap()] {
                assert!(
                    actual
                        .iter()
                        .zip(&expected[&output].values)
                        .all(|(&value, &reference)| value.is_finite()
                            && within_tolerance(value, reference, 1e-4, 1e-4))
                );
            }
        }

        // Two formerly failing 1x1 dispatches share one device and submission.
        let mut graph = Graph::new();
        let left = graph.input(TensorSpec::f32([1, 1]).unwrap());
        let right = graph.input(TensorSpec::f32([1, 1]).unwrap());
        let first = graph.matmul(left, right).unwrap();
        let second = graph.matmul(left, right).unwrap();
        graph.set_outputs(vec![first, second]).unwrap();
        let inputs = Inputs::from([(left, vec![2.]), (right, vec![-3.])]);
        let actual = execute_graph_with_kernel(&graph, &inputs, MatMulKernel::Auto).unwrap();
        assert_eq!(actual.outputs[&first], [-6.]);
        assert_eq!(actual.outputs[&second], [-6.]);

        let mut graph = Graph::new();
        let input = graph.input(TensorSpec::f32([2, 3]).unwrap());
        let bias = graph.input(TensorSpec::f32([3]).unwrap());
        let biased = graph.bias_add(input, bias).unwrap();
        let relu = graph.relu(biased).unwrap();
        let second_branch = graph.relu(input).unwrap();
        graph
            .set_outputs(vec![input, biased, relu, second_branch])
            .unwrap();
        let inputs = Inputs::from([
            (input, vec![-2., 0., 2., 4., -4., 1.]),
            (bias, vec![1., 0., -3.]),
        ]);
        let actual = execute_graph(&graph, &inputs).unwrap();
        assert_eq!(actual.outputs[&input], [-2., 0., 2., 4., -4., 1.]);
        assert_eq!(actual.outputs[&biased], [-1., 0., -1., 5., -4., -2.]);
        assert_eq!(actual.outputs[&relu], [0., 0., 0., 5., 0., 0.]);
        assert_eq!(actual.outputs[&second_branch], [0., 0., 2., 4., 0., 1.]);
    }
}

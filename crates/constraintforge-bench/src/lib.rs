use std::{
    fs::{self, OpenOptions},
    hash::Hasher,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use constraintforge_gpu::{
    AdapterSummary, MatMulKernelVariant, PreparedMatmulBenchmark, PreparedSample,
};
use constraintforge_ir::TensorSpec;
use constraintforge_reference::{Tensor, matmul, within_tolerance};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const SCHEMA_VERSION: &str = "constraintforge-benchmark-v1";
pub const ABSOLUTE_TOLERANCE: f32 = 1e-5;
pub const RELATIVE_TOLERANCE: f32 = 1e-4;
const MAX_RUNS: u32 = 10_000;
const MAX_TENSOR_BYTES: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelChoice {
    Naive,
    Tiled,
    Both,
}

#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub shape: [usize; 3],
    pub warmups: u32,
    pub runs: u32,
    pub seed: u64,
    pub kernel: KernelChoice,
}

impl BenchmarkConfig {
    pub fn validate(&self) -> Result<(), BenchError> {
        if self.shape.contains(&0) {
            return Err(BenchError::InvalidArgument(
                "M, K, and N must all be nonzero",
            ));
        }
        if self.runs == 0 || self.runs > MAX_RUNS || self.warmups > MAX_RUNS {
            return Err(BenchError::InvalidArgument(
                "runs must be 1..=10000 and warmups must be 0..=10000",
            ));
        }
        let [m, k, n] = self.shape;
        for elements in [m.checked_mul(k), k.checked_mul(n), m.checked_mul(n)] {
            let bytes = elements
                .and_then(|value| value.checked_mul(size_of::<f32>()))
                .and_then(|value| u64::try_from(value).ok())
                .ok_or(BenchError::InvalidArgument("tensor size overflows"))?;
            if bytes > MAX_TENSOR_BYTES {
                return Err(BenchError::InvalidArgument(
                    "each benchmark tensor is limited to 1 GiB",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Statistics {
    pub count: usize,
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    pub median: f64,
    pub p95_nearest_rank: Option<f64>,
    pub sample_standard_deviation: Option<f64>,
}

impl Statistics {
    pub fn calculate(samples: &[f64]) -> Result<Self, BenchError> {
        if samples.is_empty() {
            return Err(BenchError::InvalidSamples("sample set is empty"));
        }
        if samples
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(BenchError::InvalidSamples(
                "samples must be finite and nonnegative",
            ));
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        let p95_nearest_rank = (count >= 2).then(|| {
            let rank = (0.95 * count as f64).ceil() as usize;
            sorted[rank.saturating_sub(1)]
        });
        let sample_standard_deviation = (count >= 2).then(|| {
            let squared_error: f64 = sorted.iter().map(|value| (value - mean).powi(2)).sum();
            (squared_error / (count - 1) as f64).sqrt()
        });
        Ok(Self {
            count,
            minimum: sorted[0],
            maximum: sorted[count - 1],
            mean,
            median,
            p95_nearest_rank,
            sample_standard_deviation,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Correctness {
    pub passed: bool,
    pub maximum_absolute_error: f32,
    pub maximum_relative_error: f32,
    pub mismatch_count: usize,
    pub first_mismatch: Option<Mismatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mismatch {
    pub index: usize,
    pub actual: f32,
    pub expected: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelResult {
    pub variant: String,
    pub tile: [u32; 3],
    pub workgroup: [u32; 3],
    pub workgroup_memory_bytes: u32,
    pub dispatch: [u32; 3],
    pub raw_host_prepared_end_to_end_ns: Vec<u64>,
    pub raw_gpu_compute_pass_ns: Option<Vec<f64>>,
    pub host_statistics_ns: Statistics,
    pub gpu_statistics_ns: Option<Statistics>,
    pub correctness: Correctness,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub schema_version: String,
    pub run_id: String,
    pub utc_unix_seconds: u64,
    pub benchmark_name: String,
    pub shape_mkn: [usize; 3],
    pub dtype: String,
    pub input_seed: u64,
    pub hardware: HardwareMetadata,
    pub software: SoftwareMetadata,
    pub protocol: ProtocolMetadata,
    pub preparation_wall_time_ns: u64,
    pub kernels: Vec<KernelResult>,
    pub errors: Vec<String>,
    pub complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkSuiteResult {
    pub schema_version: String,
    pub suite_name: String,
    pub results: Vec<BenchmarkResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareMetadata {
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub timestamp_query_supported: bool,
    pub timestamp_period_ns: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftwareMetadata {
    pub rust_version: Option<String>,
    pub cargo_version: Option<String>,
    pub wgpu_version: String,
    pub operating_system: String,
    pub architecture: String,
    pub build_profile: String,
    pub source_revision: String,
    pub git_worktree_clean: Option<bool>,
    pub source_fingerprint_fnv1a64: String,
    pub fingerprint_paths: Vec<String>,
    #[serde(default)]
    pub source_tree_sha256: Option<String>,
    #[serde(default)]
    pub source_manifest_sha256: Vec<SourceFileHash>,
    #[serde(default)]
    pub cargo_lock_sha256: Option<String>,
    #[serde(default)]
    pub rust_toolchain_sha256: Option<String>,
    #[serde(default)]
    pub research_implementation_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceFileHash {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolMetadata {
    pub warmups_per_kernel: u32,
    pub measured_repetitions_per_kernel: u32,
    pub measurement_method: String,
    pub execution_order: Vec<String>,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub pipelines_precompiled: bool,
    pub buffers_reused: bool,
    pub output_readback_inside_host_timing: bool,
    pub query_resolution_outside_compute_pass: bool,
}

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("invalid benchmark argument: {0}")]
    InvalidArgument(&'static str),
    #[error("invalid benchmark samples: {0}")]
    InvalidSamples(&'static str),
    #[error("benchmark correctness failed for {0}")]
    Correctness(String),
    #[error(transparent)]
    Gpu(#[from] constraintforge_gpu::GpuError),
    #[error(transparent)]
    Ir(#[from] constraintforge_ir::IrError),
    #[error(transparent)]
    Reference(#[from] constraintforge_reference::ReferenceError),
    #[error("benchmark result I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("benchmark result serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
}

pub fn deterministic_inputs(
    shape: [usize; 3],
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>), BenchError> {
    let [m, k, n] = shape;
    let left_len = m
        .checked_mul(k)
        .ok_or(BenchError::InvalidArgument("left tensor size overflows"))?;
    let right_len = k
        .checked_mul(n)
        .ok_or(BenchError::InvalidArgument("right tensor size overflows"))?;
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
        unit * 2.0 - 1.0
    };
    Ok((
        (0..left_len).map(|_| next()).collect(),
        (0..right_len).map(|_| next()).collect(),
    ))
}

pub fn comparison_order(runs: u32, seed: u64) -> Vec<MatMulKernelVariant> {
    let first = if seed & 1 == 0 {
        MatMulKernelVariant::Naive
    } else {
        MatMulKernelVariant::Tiled
    };
    (0..runs.saturating_mul(2))
        .map(|index| if index % 2 == 0 { first } else { other(first) })
        .collect()
}

pub fn run(config: &BenchmarkConfig) -> Result<BenchmarkResult, BenchError> {
    config.validate()?;
    let (left, right) = deterministic_inputs(config.shape, config.seed)?;
    let [m, k, n] = config.shape;
    let reference = matmul(
        &Tensor::new(TensorSpec::f32([m, k])?, left.clone())?,
        &Tensor::new(TensorSpec::f32([k, n])?, right.clone())?,
    )?;
    let prepared = PreparedMatmulBenchmark::prepare(config.shape, &left, &right)?;
    let variants = match config.kernel {
        KernelChoice::Naive => vec![MatMulKernelVariant::Naive],
        KernelChoice::Tiled => vec![MatMulKernelVariant::Tiled],
        KernelChoice::Both => vec![MatMulKernelVariant::Naive, MatMulKernelVariant::Tiled],
    };
    let mut correctness = Vec::new();
    for &variant in &variants {
        let sample = prepared.invoke(variant)?;
        let result = compare_output(&sample.output, &reference.values);
        if !result.passed {
            return Err(BenchError::Correctness(variant_name(variant).into()));
        }
        correctness.push((variant, result));
    }
    let warmup_order = execution_order(config.warmups, config.seed, &variants);
    for variant in warmup_order {
        prepared.invoke(variant)?;
    }
    let order = execution_order(config.runs, config.seed, &variants);
    let mut samples: Vec<(MatMulKernelVariant, Vec<PreparedSample>)> = variants
        .iter()
        .copied()
        .map(|variant| (variant, Vec::with_capacity(config.runs as usize)))
        .collect();
    for variant in &order {
        let sample = prepared.invoke(*variant)?;
        samples
            .iter_mut()
            .find(|(candidate, _)| candidate == variant)
            .expect("prepared sample bucket exists")
            .1
            .push(sample);
    }
    let mut kernels = Vec::with_capacity(samples.len());
    for (variant, variant_samples) in samples {
        let metadata = prepared.kernel_metadata(variant)?;
        let raw_host: Vec<u64> = variant_samples
            .iter()
            .map(|sample| sample.host_duration_ns)
            .collect();
        let gpu_available = variant_samples
            .iter()
            .all(|sample| sample.gpu_compute_pass_duration_ns.is_some());
        let raw_gpu = gpu_available.then(|| {
            variant_samples
                .iter()
                .filter_map(|sample| sample.gpu_compute_pass_duration_ns)
                .collect::<Vec<_>>()
        });
        kernels.push(KernelResult {
            variant: variant_name(variant).into(),
            tile: metadata.config.tile(),
            workgroup: metadata.config.workgroup(),
            workgroup_memory_bytes: metadata.config.shared_memory_bytes()?,
            dispatch: metadata.dispatch,
            host_statistics_ns: Statistics::calculate(
                &raw_host.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            )?,
            gpu_statistics_ns: raw_gpu.as_deref().map(Statistics::calculate).transpose()?,
            raw_host_prepared_end_to_end_ns: raw_host,
            raw_gpu_compute_pass_ns: raw_gpu,
            correctness: correctness
                .iter()
                .find(|(candidate, _)| *candidate == variant)
                .expect("correctness result exists")
                .1
                .clone(),
        });
    }
    let now = unix_seconds()?;
    let run_nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| BenchError::InvalidArgument("system clock precedes the Unix epoch"))?
        .as_nanos();
    let software = current_software_metadata()?;
    Ok(BenchmarkResult {
        schema_version: SCHEMA_VERSION.into(),
        run_id: format!("{run_nonce}-{}", std::process::id()),
        utc_unix_seconds: now,
        benchmark_name: "prepared-matmul".into(),
        shape_mkn: config.shape,
        dtype: "F32".into(),
        input_seed: config.seed,
        hardware: hardware_metadata(prepared.adapter(), prepared.timestamp_period_ns()),
        software,
        protocol: ProtocolMetadata {
            warmups_per_kernel: config.warmups,
            measured_repetitions_per_kernel: config.runs,
            measurement_method: "one synchronized prepared invocation at a time; host includes encoding, submission, compute, synchronization, timestamp/output readback; GPU timestamp spans the compute pass only".into(),
            execution_order: order.into_iter().map(|variant| variant_name(variant).into()).collect(),
            absolute_tolerance: ABSOLUTE_TOLERANCE,
            relative_tolerance: RELATIVE_TOLERANCE,
            pipelines_precompiled: true,
            buffers_reused: true,
            output_readback_inside_host_timing: true,
            query_resolution_outside_compute_pass: true,
        },
        preparation_wall_time_ns: u64::try_from(prepared.preparation_duration().as_nanos()).unwrap_or(u64::MAX),
        kernels,
        errors: Vec::new(),
        complete: true,
    })
}

pub fn current_software_metadata() -> Result<SoftwareMetadata, BenchError> {
    let (source_revision, clean, fingerprint, fingerprint_paths) = provenance(project_root())?;
    let (source_tree_sha256, source_manifest_sha256) = source_sha256_manifest(project_root())?;
    Ok(SoftwareMetadata {
        rust_version: command_version("rustc"),
        cargo_version: command_version("cargo"),
        wgpu_version: "30.0.1".into(),
        operating_system: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
        source_revision,
        git_worktree_clean: clean,
        source_fingerprint_fnv1a64: fingerprint,
        fingerprint_paths,
        source_tree_sha256: Some(source_tree_sha256),
        cargo_lock_sha256: file_sha256(&project_root().join("Cargo.lock"))?,
        rust_toolchain_sha256: file_sha256(&project_root().join("rust-toolchain.toml"))?,
        source_manifest_sha256,
        research_implementation_version: Some("m6-v1".into()),
    })
}

pub fn write_result(path: &Path, result: &BenchmarkResult) -> Result<(), BenchError> {
    write_json(path, result)
}

pub fn write_suite(path: &Path, result: &BenchmarkSuiteResult) -> Result<(), BenchError> {
    write_json(path, result)
}

fn write_json(path: &Path, result: &impl Serialize) -> Result<(), BenchError> {
    if path.as_os_str().is_empty() {
        return Err(BenchError::InvalidArgument("output path must not be empty"));
    }
    if path.exists() {
        return Err(BenchError::OutputExists(path.to_path_buf()));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .ok_or(BenchError::InvalidArgument("output path must name a file"))?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let write_result = (|| -> Result<(), BenchError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, result)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                BenchError::OutputExists(path.to_path_buf())
            } else {
                BenchError::Io(error)
            }
        })?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    write_result
}

fn execution_order(
    runs: u32,
    seed: u64,
    variants: &[MatMulKernelVariant],
) -> Vec<MatMulKernelVariant> {
    if variants.len() == 2 {
        comparison_order(runs, seed)
    } else {
        vec![variants[0]; runs as usize]
    }
}

fn other(variant: MatMulKernelVariant) -> MatMulKernelVariant {
    match variant {
        MatMulKernelVariant::Naive => MatMulKernelVariant::Tiled,
        MatMulKernelVariant::Tiled => MatMulKernelVariant::Naive,
    }
}

fn variant_name(variant: MatMulKernelVariant) -> &'static str {
    match variant {
        MatMulKernelVariant::Naive => "naive",
        MatMulKernelVariant::Tiled => "tiled",
    }
}

fn compare_output(actual: &[f32], expected: &[f32]) -> Correctness {
    let mut maximum_absolute_error = 0.0f32;
    let mut maximum_relative_error = 0.0f32;
    let mut mismatch_count = actual.len().abs_diff(expected.len());
    let mut first_mismatch = None;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(ABSOLUTE_TOLERANCE);
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        if !actual.is_finite()
            || !expected.is_finite()
            || !within_tolerance(actual, expected, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
        {
            mismatch_count += 1;
            first_mismatch.get_or_insert(Mismatch {
                index,
                actual,
                expected,
            });
        }
    }
    Correctness {
        passed: mismatch_count == 0,
        maximum_absolute_error,
        maximum_relative_error,
        mismatch_count,
        first_mismatch,
    }
}

fn hardware_metadata(
    adapter: &AdapterSummary,
    timestamp_period_ns: Option<f32>,
) -> HardwareMetadata {
    HardwareMetadata {
        adapter_name: adapter.name.clone(),
        backend: adapter.backend.clone(),
        device_type: adapter.device_type.clone(),
        max_buffer_size: adapter.max_buffer_size,
        max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: adapter.max_compute_workgroups_per_dimension,
        timestamp_query_supported: timestamp_period_ns.is_some(),
        timestamp_period_ns,
    }
}

fn command_version(program: &str) -> Option<String> {
    Command::new(program)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
}

fn unix_seconds() -> Result<u64, BenchError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| BenchError::InvalidArgument("system clock precedes the Unix epoch"))
}

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("benchmark crate is inside workspace/crates")
}

fn provenance(root: &Path) -> Result<(String, Option<bool>, String, Vec<String>), BenchError> {
    let git_dir = root.join(".git");
    let (revision, clean) = if git_dir.exists() {
        let revision = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(root)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|| "unavailable".into());
        let clean = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(root)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| output.stdout.is_empty());
        (revision, clean)
    } else {
        ("unavailable/unversioned".into(), None)
    };
    let (fingerprint, paths) = source_fingerprint(root)?;
    Ok((revision, clean, fingerprint, paths))
}

fn source_fingerprint(root: &Path) -> Result<(String, Vec<String>), BenchError> {
    let mut paths = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
    ];
    collect_sources(&root.join("crates"), &mut paths)?;
    paths.retain(|path| path.is_file());
    paths.sort();
    let mut hasher = Fnv1a64::default();
    let mut relative = Vec::with_capacity(paths.len());
    for path in paths {
        let name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        hasher.write(name.as_bytes());
        hasher.write(&[0]);
        hasher.write(&fs::read(&path)?);
        hasher.write(&[0xff]);
        relative.push(name);
    }
    Ok((format!("{:016x}", hasher.finish()), relative))
}

fn source_sha256_manifest(root: &Path) -> Result<(String, Vec<SourceFileHash>), BenchError> {
    let mut paths = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
    ];
    collect_sources(&root.join("crates"), &mut paths)?;
    collect_sources(&root.join("research/protocols"), &mut paths)?;
    paths.retain(|path| path.is_file());
    paths.sort();
    let mut tree = Sha256::new();
    let mut manifest = Vec::with_capacity(paths.len());
    for path in paths {
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(&path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        tree.update(relative.as_bytes());
        tree.update([0]);
        tree.update(digest.as_bytes());
        tree.update([0xff]);
        manifest.push(SourceFileHash {
            path: relative,
            sha256: digest,
        });
    }
    Ok((format!("{:x}", tree.finalize()), manifest))
}

fn file_sha256(path: &Path) -> Result<Option<String>, BenchError> {
    path.is_file()
        .then(|| fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
        .transpose()
        .map_err(BenchError::Io)
}

fn collect_sources(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<(), BenchError> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_sources(&path, paths)?;
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("rs" | "toml" | "wgsl")
        ) {
            paths.push(path);
        }
    }
    Ok(())
}

#[derive(Default)]
struct Fnv1a64(u64);

impl Hasher for Fnv1a64 {
    fn finish(&self) -> u64 {
        if self.0 == 0 {
            0xcbf29ce484222325
        } else {
            self.0
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.finish();
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        self.0 = hash;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            shape: [2, 2, 2],
            warmups: 1,
            runs: 3,
            seed: 42,
            kernel: KernelChoice::Both,
        }
    }

    #[test]
    fn validates_arguments_and_deterministic_inputs() {
        config().validate().unwrap();
        let mut invalid = config();
        invalid.runs = 0;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.shape[0] = 0;
        assert!(invalid.validate().is_err());
        assert_eq!(
            deterministic_inputs([2, 3, 4], 7).unwrap(),
            deterministic_inputs([2, 3, 4], 7).unwrap()
        );
        assert_ne!(
            deterministic_inputs([2, 3, 4], 7).unwrap(),
            deterministic_inputs([2, 3, 4], 8).unwrap()
        );
    }

    #[test]
    fn calculates_documented_statistics() {
        let stats = Statistics::calculate(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.minimum, 1.0);
        assert_eq!(stats.maximum, 4.0);
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.median, 2.5);
        assert_eq!(stats.p95_nearest_rank, Some(4.0));
        assert!((stats.sample_standard_deviation.unwrap() - 1.290_994_448_735_805_6).abs() < 1e-12);
        let one = Statistics::calculate(&[2.0]).unwrap();
        assert_eq!(one.p95_nearest_rank, None);
        assert_eq!(one.sample_standard_deviation, None);
        assert!(Statistics::calculate(&[]).is_err());
        assert!(Statistics::calculate(&[f64::NAN]).is_err());
        assert!(Statistics::calculate(&[-1.0]).is_err());
    }

    #[test]
    fn comparison_order_is_balanced_and_deterministic() {
        let order = comparison_order(3, 42);
        assert_eq!(order.len(), 6);
        assert_eq!(order, comparison_order(3, 42));
        assert_eq!(
            order
                .iter()
                .filter(|&&value| value == MatMulKernelVariant::Naive)
                .count(),
            3
        );
        assert_eq!(
            order
                .iter()
                .filter(|&&value| value == MatMulKernelVariant::Tiled)
                .count(),
            3
        );
        assert_ne!(order, comparison_order(3, 43));
    }

    #[test]
    fn serializes_schema_and_refuses_overwrite() {
        let result = BenchmarkResult {
            schema_version: SCHEMA_VERSION.into(),
            run_id: "test".into(),
            utc_unix_seconds: 0,
            benchmark_name: "test".into(),
            shape_mkn: [2, 2, 2],
            dtype: "F32".into(),
            input_seed: 1,
            hardware: HardwareMetadata {
                adapter_name: "test".into(),
                backend: "test".into(),
                device_type: "test".into(),
                max_buffer_size: 1,
                max_storage_buffer_binding_size: 1,
                max_compute_workgroups_per_dimension: 1,
                timestamp_query_supported: false,
                timestamp_period_ns: None,
            },
            software: SoftwareMetadata {
                rust_version: None,
                cargo_version: None,
                wgpu_version: "30.0.1".into(),
                operating_system: "test".into(),
                architecture: "test".into(),
                build_profile: "test".into(),
                source_revision: "unavailable/unversioned".into(),
                git_worktree_clean: None,
                source_fingerprint_fnv1a64: "0".into(),
                fingerprint_paths: vec![],
                source_tree_sha256: None,
                source_manifest_sha256: vec![],
                cargo_lock_sha256: None,
                rust_toolchain_sha256: None,
                research_implementation_version: None,
            },
            protocol: ProtocolMetadata {
                warmups_per_kernel: 0,
                measured_repetitions_per_kernel: 1,
                measurement_method: "test".into(),
                execution_order: vec!["naive".into()],
                absolute_tolerance: ABSOLUTE_TOLERANCE,
                relative_tolerance: RELATIVE_TOLERANCE,
                pipelines_precompiled: true,
                buffers_reused: true,
                output_readback_inside_host_timing: true,
                query_resolution_outside_compute_pass: true,
            },
            preparation_wall_time_ns: 1,
            kernels: vec![],
            errors: vec![],
            complete: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains(SCHEMA_VERSION));
        assert!(json.contains("unavailable/unversioned"));
        let path = std::env::temp_dir().join(format!(
            "constraintforge-bench-test-{}.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        write_result(&path, &result).unwrap();
        assert!(matches!(
            write_result(&path, &result),
            Err(BenchError::OutputExists(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_fingerprint_is_stable_and_complete() {
        let (first, paths) = source_fingerprint(project_root()).unwrap();
        let (second, _) = source_fingerprint(project_root()).unwrap();
        assert_eq!(first, second);
        assert!(paths.contains(&"Cargo.toml".into()));
        assert!(
            paths
                .iter()
                .any(|path| path.ends_with("constraintforge-gpu/src/lib.rs"))
        );
        let (revision, clean, _, _) = provenance(project_root()).unwrap();
        if project_root().join(".git").exists() {
            assert_eq!(revision.len(), 40);
            assert!(clean.is_some());
        } else {
            assert_eq!(revision, "unavailable/unversioned");
            assert_eq!(clean, None);
        }
    }

    #[test]
    fn source_sha256_manifest_is_stable_relative_and_complete() {
        let first = source_sha256_manifest(project_root()).unwrap();
        let second = source_sha256_manifest(project_root()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.0.len(), 64);
        assert!(
            first
                .1
                .iter()
                .all(|entry| { !Path::new(&entry.path).is_absolute() && entry.sha256.len() == 64 })
        );
        assert!(first.1.iter().any(|entry| entry.path == "Cargo.lock"));
        assert!(
            first
                .1
                .iter()
                .any(|entry| entry.path == "research/protocols/m6-v1.toml")
        );
    }
}

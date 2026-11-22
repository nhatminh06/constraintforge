use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use constraintforge_bench::{
    ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, SoftwareMetadata, Statistics,
    current_software_metadata, deterministic_inputs,
};
use constraintforge_gpu::{
    AdapterSummary, KernelCandidateId, MatMulKernelConfig, PreparedMatmulBenchmark,
    legal_tiled_matmul_candidates, tiled_matmul_candidates,
};
use constraintforge_ir::TensorSpec;
use constraintforge_reference::{Tensor, matmul, within_tolerance};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const TUNING_SCHEMA_VERSION: &str = "constraintforge-tuning-v1";
pub const POLICY_VERSION: &str = "m5-v1";
pub const RESOURCE_RANKED_POLICY_VERSION: &str = "m6-resource-ranked-v1";
pub const EXHAUSTIVE_POLICY_VERSION: &str = "m6-exhaustive-v1";
pub const DEVELOPMENT_SHAPES: [[usize; 3]; 6] = [
    [2, 2, 2],
    [15, 17, 13],
    [31, 33, 29],
    [64, 64, 64],
    [128, 128, 128],
    [256, 128, 192],
];
pub const HELD_OUT_SHAPES: [[usize; 3]; 5] = [
    [7, 19, 11],
    [48, 96, 80],
    [96, 64, 160],
    [127, 129, 131],
    [256, 64, 384],
];

pub fn tiled_candidate_ids() -> Vec<String> {
    tiled_matmul_candidates()
        .into_iter()
        .map(candidate_id)
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Policy {
    Fixed,
    Heuristic,
    Random,
    ResourceRankedOnly,
    ResourceAware,
    Exhaustive,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementMode {
    GpuTimestamp,
    HostPreparedEndToEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScoreAblation {
    Full,
    NoPadding,
    NoShared,
    NoReuse,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchDesign {
    #[default]
    Legacy,
    MatchedBatchesV2,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchBudget {
    pub max_candidates: usize,
    pub max_batches: usize,
    pub max_duration: Duration,
}

impl SearchBudget {
    pub fn validate(self) -> Result<(), TunerError> {
        if self.max_candidates > 10_000 || self.max_batches > 100_000 {
            return Err(TunerError::InvalidArgument(
                "search budget is unreasonably large",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BatchObservation {
    pub samples_ns: Vec<f64>,
    pub correctness_passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub id: String,
    pub variant: String,
    pub tile: [u32; 3],
    pub workgroup: [u32; 3],
    pub shader_variant: u32,
    pub shared_memory_bytes: u32,
    pub dispatch: [u32; 3],
    pub padded_output_ratio: f64,
    pub padded_reduction_ratio: f64,
    pub reuse_proxy: f64,
    pub static_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchDecision {
    pub batch_index: usize,
    pub candidate_id: String,
    pub reason: String,
    pub raw_samples_ns: Vec<f64>,
    pub median_ns: Option<f64>,
    pub correctness_passed: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchOutcome {
    pub policy: Policy,
    pub selected_candidate_id: String,
    pub selected_score_ns: Option<f64>,
    pub candidates_considered: usize,
    pub candidates_measured: usize,
    pub measurement_batches: usize,
    pub total_measured_dispatches: usize,
    pub search_decision_overhead_ns: u64,
    pub ordered_history: Vec<SearchDecision>,
    pub termination_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectnessAdmission {
    pub candidate_id: String,
    pub passed: bool,
    pub maximum_absolute_error: f32,
    pub maximum_relative_error: f32,
    pub mismatch_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuningResult {
    pub schema_version: String,
    pub trial_id: String,
    pub utc_unix_seconds: u64,
    pub shape_mkn: [usize; 3],
    pub input_seed: u64,
    pub search_seed: u64,
    pub policy_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_design: Option<SearchDesign>,
    pub measurement_mode: MeasurementMode,
    pub budget_max_candidates: usize,
    pub budget_max_batches: usize,
    pub budget_max_duration_ms: u64,
    pub adapter: AdapterSummaryRecord,
    pub software: SoftwareMetadata,
    pub candidate_pool: Vec<CandidateRecord>,
    pub static_rejections: Vec<String>,
    pub correctness_admission: Vec<CorrectnessAdmission>,
    pub search: SearchOutcome,
    pub independent_evaluation_samples_ns: Vec<f64>,
    pub independent_evaluation_statistics_ns: Statistics,
    pub preparation_time_ns: u64,
    pub correctness_validation_time_ns: u64,
    pub search_wall_time_ns: u64,
    pub tuning_wall_time_ns: u64,
    pub search_decision_overhead_ns: u64,
    pub compiled_candidates: usize,
    pub complete: bool,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuningComparisonResult {
    pub schema_version: String,
    pub results: Vec<TuningResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterSummaryRecord {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub timestamp_query_supported: bool,
    pub timestamp_period_ns: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct GpuTuningConfig {
    pub shape: [usize; 3],
    pub workload_seed: u64,
    pub search_seed: u64,
    pub policy: Policy,
    pub score_ablation: ScoreAblation,
    pub search_design: SearchDesign,
    pub budget: SearchBudget,
    pub measurement_mode: MeasurementMode,
    pub tuning_warmups: u32,
    pub tuning_runs: u32,
    pub evaluation_warmups: u32,
    pub evaluation_runs: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceBlockResult {
    pub block_index: usize,
    pub order_index: usize,
    pub candidate_id: String,
    pub started_utc: u64,
    pub ended_utc: u64,
    pub raw_samples_ns: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct GpuReferenceBatchConfig {
    pub shape: [usize; 3],
    pub workload_seed: u64,
    pub candidate_id: String,
    pub block_index: usize,
    pub order_index: usize,
    pub warmups: u32,
    pub samples: u32,
}

#[derive(Debug, Error)]
pub enum TunerError {
    #[error("invalid tuner argument: {0}")]
    InvalidArgument(&'static str),
    #[error("no valid candidate was measured")]
    NoValidCandidate,
    #[error("GPU timestamps are unavailable; select host-prepared-end-to-end mode explicitly")]
    TimestampUnavailable,
    #[error("candidate correctness failed: {0}")]
    Correctness(String),
    #[error(transparent)]
    Gpu(#[from] constraintforge_gpu::GpuError),
    #[error(transparent)]
    Bench(#[from] constraintforge_bench::BenchError),
    #[error(transparent)]
    Ir(#[from] constraintforge_ir::IrError),
    #[error(transparent)]
    Reference(#[from] constraintforge_reference::ReferenceError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
}

pub fn candidate_records(shape: [usize; 3]) -> Result<Vec<CandidateRecord>, TunerError> {
    let mut seen = HashSet::new();
    tiled_matmul_candidates()
        .into_iter()
        .filter(|config| seen.insert(config.candidate_id()))
        .map(|config| candidate_record(config, shape))
        .collect()
}

pub fn schedule<F>(
    candidates: &[CandidateRecord],
    policy: Policy,
    budget: SearchBudget,
    seed: u64,
    measure: F,
) -> Result<SearchOutcome, TunerError>
where
    F: FnMut(&CandidateRecord) -> Result<BatchObservation, String>,
{
    schedule_with_design(
        candidates,
        policy,
        budget,
        seed,
        SearchDesign::Legacy,
        measure,
    )
}

pub fn schedule_with_design<F>(
    candidates: &[CandidateRecord],
    policy: Policy,
    budget: SearchBudget,
    seed: u64,
    design: SearchDesign,
    mut measure: F,
) -> Result<SearchOutcome, TunerError>
where
    F: FnMut(&CandidateRecord) -> Result<BatchObservation, String>,
{
    budget.validate()?;
    let decision_started = Instant::now();
    if candidates.is_empty() {
        return Err(TunerError::NoValidCandidate);
    }
    if policy == Policy::Fixed {
        let selected = candidates
            .iter()
            .find(|candidate| candidate.tile == [16, 16, 16])
            .unwrap_or(&candidates[0]);
        return Ok(unmeasured_outcome(
            policy,
            selected,
            candidates.len(),
            "fixed baseline",
            duration_ns(decision_started.elapsed()),
        ));
    }
    if policy == Policy::Heuristic {
        let selected = candidates
            .iter()
            .min_by(|left, right| left.static_score.total_cmp(&right.static_score))
            .ok_or(TunerError::NoValidCandidate)?;
        return Ok(unmeasured_outcome(
            policy,
            selected,
            candidates.len(),
            "deterministic static heuristic",
            duration_ns(decision_started.elapsed()),
        ));
    }
    if budget.max_candidates == 0 || budget.max_batches == 0 || budget.max_duration.is_zero() {
        return Err(TunerError::NoValidCandidate);
    }

    let started = Instant::now();
    let mut ordered: Vec<_> = (0..candidates.len()).collect();
    match policy {
        Policy::Random => seeded_shuffle(&mut ordered, seed),
        Policy::ResourceRankedOnly | Policy::ResourceAware | Policy::Exhaustive => {
            ordered.sort_by(|&left, &right| {
                candidates[left]
                    .static_score
                    .total_cmp(&candidates[right].static_score)
            })
        }
        Policy::Fixed | Policy::Heuristic => unreachable!(),
    }
    ordered.truncate(budget.max_candidates.min(ordered.len()));
    let mut decision_overhead_ns = duration_ns(decision_started.elapsed());
    let mut history = Vec::new();
    let mut scores: Vec<(usize, Vec<f64>)> = Vec::new();
    for &candidate_index in &ordered {
        if history.len() >= budget.max_batches || started.elapsed() >= budget.max_duration {
            break;
        }
        record_batch(
            &candidates[candidate_index],
            &mut history,
            &mut scores,
            candidate_index,
            "initial distinct-candidate measurement",
            &mut measure,
        );
    }
    if matches!(policy, Policy::Random | Policy::ResourceRankedOnly)
        && design == SearchDesign::MatchedBatchesV2
        && !ordered.is_empty()
    {
        let mut repeat = 0;
        while history.len() < budget.max_batches && started.elapsed() < budget.max_duration {
            let candidate_index = ordered[repeat % ordered.len()];
            record_batch(
                &candidates[candidate_index],
                &mut history,
                &mut scores,
                candidate_index,
                "prespecified nonadaptive round-robin repeat",
                &mut measure,
            );
            repeat += 1;
        }
    }
    if policy == Policy::ResourceAware {
        while history.len() < budget.max_batches && started.elapsed() < budget.max_duration {
            let decision = Instant::now();
            let best = best_score(&scores);
            decision_overhead_ns =
                decision_overhead_ns.saturating_add(duration_ns(decision.elapsed()));
            let Some((best_index, _)) = best else {
                break;
            };
            record_batch(
                &candidates[best_index],
                &mut history,
                &mut scores,
                best_index,
                "repeat current best after static-priority exploration",
                &mut measure,
            );
        }
    }
    let decision = Instant::now();
    let best = best_score(&scores);
    decision_overhead_ns = decision_overhead_ns.saturating_add(duration_ns(decision.elapsed()));
    let (selected_index, selected_score) = best.ok_or(TunerError::NoValidCandidate)?;
    let measured: HashSet<_> = history
        .iter()
        .filter(|decision| decision.error.is_none())
        .map(|decision| decision.candidate_id.as_str())
        .collect();
    let termination_reason = if started.elapsed() >= budget.max_duration {
        "time budget reached"
    } else if history.len() >= budget.max_batches {
        "measurement-batch budget reached"
    } else {
        "candidate budget exhausted"
    };
    Ok(SearchOutcome {
        policy,
        selected_candidate_id: candidates[selected_index].id.clone(),
        selected_score_ns: Some(selected_score),
        candidates_considered: candidates.len(),
        candidates_measured: measured.len(),
        measurement_batches: history.len(),
        total_measured_dispatches: history
            .iter()
            .map(|decision| decision.raw_samples_ns.len())
            .sum(),
        search_decision_overhead_ns: decision_overhead_ns,
        ordered_history: history,
        termination_reason: termination_reason.into(),
    })
}

pub fn run_gpu(config: &GpuTuningConfig) -> Result<TuningResult, TunerError> {
    config.budget.validate()?;
    if config.tuning_runs == 0 || config.evaluation_runs == 0 {
        return Err(TunerError::InvalidArgument(
            "tuning and evaluation runs must be nonzero",
        ));
    }
    let total_started = Instant::now();
    let (left, right) = deterministic_inputs(config.shape, config.workload_seed)?;
    let [m, k, n] = config.shape;
    let reference = matmul(
        &Tensor::new(TensorSpec::f32([m, k])?, left.clone())?,
        &Tensor::new(TensorSpec::f32([k, n])?, right.clone())?,
    )?;
    let (_, all_configs, rejections) = legal_tiled_matmul_candidates(config.shape)?;
    if all_configs.is_empty() {
        return Err(TunerError::NoValidCandidate);
    }
    let legal_ids = all_configs
        .iter()
        .map(|kernel| candidate_id(*kernel))
        .collect::<HashSet<_>>();
    let mut records = candidate_records(config.shape)?
        .into_iter()
        .filter(|record| legal_ids.contains(&record.id))
        .collect::<Vec<_>>();
    for record in &mut records {
        record.static_score = match config.score_ablation {
            ScoreAblation::Full => {
                record.padded_output_ratio * record.padded_reduction_ratio
                    + f64::from(record.shared_memory_bytes) / 65_536.0
                    + 1.0 / record.reuse_proxy
            }
            ScoreAblation::NoPadding => {
                f64::from(record.shared_memory_bytes) / 65_536.0 + 1.0 / record.reuse_proxy
            }
            ScoreAblation::NoShared => {
                record.padded_output_ratio * record.padded_reduction_ratio
                    + 1.0 / record.reuse_proxy
            }
            ScoreAblation::NoReuse => {
                record.padded_output_ratio * record.padded_reduction_ratio
                    + f64::from(record.shared_memory_bytes) / 65_536.0
            }
        };
    }
    let static_rejections = rejections
        .into_iter()
        .map(|rejection| format!("{:?}: {}", rejection.id, rejection.reason))
        .collect::<Vec<_>>();
    let configs = match config.policy {
        Policy::Fixed => vec![
            *all_configs
                .iter()
                .find(|kernel| **kernel == MatMulKernelConfig::tiled())
                .ok_or(TunerError::NoValidCandidate)?,
        ],
        Policy::Heuristic => {
            let selected = records
                .iter()
                .min_by(|left, right| left.static_score.total_cmp(&right.static_score))
                .ok_or(TunerError::NoValidCandidate)?;
            vec![
                *all_configs
                    .iter()
                    .find(|kernel| candidate_id(**kernel) == selected.id)
                    .ok_or(TunerError::NoValidCandidate)?,
            ]
        }
        Policy::Random
        | Policy::ResourceRankedOnly
        | Policy::ResourceAware
        | Policy::Exhaustive => all_configs,
    };
    let prepared =
        PreparedMatmulBenchmark::prepare_candidates(config.shape, &left, &right, &configs)?;
    if config.measurement_mode == MeasurementMode::GpuTimestamp
        && prepared.timestamp_period_ns().is_none()
    {
        return Err(TunerError::TimestampUnavailable);
    }
    let preparation_time_ns = duration_ns(prepared.preparation_duration());
    let correctness_started = Instant::now();
    let mut admissions = Vec::with_capacity(configs.len());
    for &kernel in &configs {
        let record = records
            .iter()
            .find(|record| record.id == candidate_id(kernel))
            .ok_or(TunerError::NoValidCandidate)?;
        let sample = prepared.invoke_config(kernel)?;
        let admission = correctness(&record.id, &sample.output, &reference.values);
        if !admission.passed {
            return Err(TunerError::Correctness(record.id.clone()));
        }
        admissions.push(admission);
    }
    let correctness_validation_time_ns = duration_ns(correctness_started.elapsed());
    let search_started = Instant::now();
    let remaining_budget = SearchBudget {
        max_duration: config
            .budget
            .max_duration
            .saturating_sub(total_started.elapsed()),
        ..config.budget
    };
    let outcome = schedule_with_design(
        &records,
        config.policy,
        remaining_budget,
        config.search_seed,
        config.search_design,
        |candidate| {
            let kernel = configs
                .iter()
                .find(|kernel| candidate_id(**kernel) == candidate.id)
                .ok_or_else(|| "candidate configuration is missing".to_owned())?;
            for _ in 0..config.tuning_warmups {
                prepared
                    .invoke_config(*kernel)
                    .map_err(|error| error.to_string())?;
            }
            let mut samples = Vec::with_capacity(config.tuning_runs as usize);
            for _ in 0..config.tuning_runs {
                let sample = prepared
                    .invoke_config(*kernel)
                    .map_err(|error| error.to_string())?;
                samples.push(sample_value(&sample, config.measurement_mode)?);
            }
            Ok(BatchObservation {
                samples_ns: samples,
                correctness_passed: true,
            })
        },
    )?;
    let search_wall_time_ns = duration_ns(search_started.elapsed());
    let tuning_wall_time_ns = duration_ns(total_started.elapsed());
    let selected = configs
        .iter()
        .find(|kernel| candidate_id(**kernel) == outcome.selected_candidate_id)
        .ok_or(TunerError::NoValidCandidate)?;
    for _ in 0..config.evaluation_warmups {
        prepared.invoke_config(*selected)?;
    }
    let mut evaluation = Vec::with_capacity(config.evaluation_runs as usize);
    for _ in 0..config.evaluation_runs {
        let sample = prepared.invoke_config(*selected)?;
        evaluation.push(
            sample_value(&sample, config.measurement_mode).map_err(TunerError::InvalidArgument)?,
        );
    }
    let evaluation_statistics = Statistics::calculate(&evaluation)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TunerError::InvalidArgument("system clock precedes Unix epoch"))?;
    let search_decision_overhead_ns = outcome.search_decision_overhead_ns;
    Ok(TuningResult {
        schema_version: TUNING_SCHEMA_VERSION.into(),
        trial_id: format!("{}-{}", now.as_nanos(), std::process::id()),
        utc_unix_seconds: now.as_secs(),
        shape_mkn: config.shape,
        input_seed: config.workload_seed,
        search_seed: config.search_seed,
        policy_version: match config.policy {
            Policy::ResourceRankedOnly => RESOURCE_RANKED_POLICY_VERSION,
            Policy::Exhaustive => EXHAUSTIVE_POLICY_VERSION,
            _ => POLICY_VERSION,
        }
        .into(),
        search_design: (config.search_design != SearchDesign::Legacy)
            .then_some(config.search_design),
        measurement_mode: config.measurement_mode,
        budget_max_candidates: config.budget.max_candidates,
        budget_max_batches: config.budget.max_batches,
        budget_max_duration_ms: config
            .budget
            .max_duration
            .as_millis()
            .min(u128::from(u64::MAX)) as u64,
        adapter: adapter_record(prepared.adapter(), prepared.timestamp_period_ns()),
        software: current_software_metadata()?,
        candidate_pool: records,
        static_rejections,
        correctness_admission: admissions,
        search: outcome,
        independent_evaluation_samples_ns: evaluation,
        independent_evaluation_statistics_ns: evaluation_statistics,
        preparation_time_ns,
        correctness_validation_time_ns,
        search_wall_time_ns,
        tuning_wall_time_ns,
        search_decision_overhead_ns,
        compiled_candidates: configs.len(),
        complete: true,
        errors: Vec::new(),
    })
}

pub fn run_gpu_reference_batch(
    config: &GpuReferenceBatchConfig,
) -> Result<
    (
        AdapterSummaryRecord,
        SoftwareMetadata,
        ReferenceBlockResult,
        u64,
        u64,
    ),
    TunerError,
> {
    if config.warmups == 0 || config.samples == 0 {
        return Err(TunerError::InvalidArgument(
            "reference warmups and samples must be nonzero",
        ));
    }
    let (left, right) = deterministic_inputs(config.shape, config.workload_seed)?;
    let (_, configs, rejections) = legal_tiled_matmul_candidates(config.shape)?;
    if !rejections.is_empty() || configs.len() != 10 {
        return Err(TunerError::InvalidArgument(
            "reference evaluation requires exactly ten legal candidates",
        ));
    }
    let kernel = configs
        .iter()
        .find(|candidate| candidate_id(**candidate) == config.candidate_id)
        .ok_or(TunerError::NoValidCandidate)?;
    let prepared =
        PreparedMatmulBenchmark::prepare_candidates(config.shape, &left, &right, &[*kernel])?;
    if prepared.timestamp_period_ns().is_none() {
        return Err(TunerError::TimestampUnavailable);
    }
    let measurement_started = Instant::now();
    for _ in 0..config.warmups {
        prepared.invoke_config(*kernel)?;
    }
    let started_utc = unix_seconds()?;
    let mut raw_samples_ns = Vec::with_capacity(config.samples as usize);
    for _ in 0..config.samples {
        raw_samples_ns.push(
            prepared
                .invoke_config(*kernel)?
                .gpu_compute_pass_duration_ns
                .ok_or(TunerError::TimestampUnavailable)?,
        );
    }
    let block = ReferenceBlockResult {
        block_index: config.block_index,
        order_index: config.order_index,
        candidate_id: config.candidate_id.clone(),
        started_utc,
        ended_utc: unix_seconds()?,
        raw_samples_ns,
    };
    Ok((
        adapter_record(prepared.adapter(), prepared.timestamp_period_ns()),
        current_software_metadata()?,
        block,
        duration_ns(prepared.preparation_duration()),
        duration_ns(measurement_started.elapsed()),
    ))
}

pub fn write_result(path: &Path, result: &TuningResult) -> Result<(), TunerError> {
    write_json(path, result)
}

pub fn write_comparison(path: &Path, result: &TuningComparisonResult) -> Result<(), TunerError> {
    write_json(path, result)
}

fn write_json(path: &Path, result: &impl Serialize) -> Result<(), TunerError> {
    if path.exists() {
        return Err(TunerError::OutputExists(path.to_path_buf()));
    }
    if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .ok_or(TunerError::InvalidArgument("output path must name a file"))?
        .to_string_lossy();
    let temporary = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    let written = (|| -> Result<(), TunerError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, result)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                TunerError::OutputExists(path.to_path_buf())
            } else {
                TunerError::Io(error)
            }
        })?;
        Ok(())
    })();
    let _ = fs::remove_file(temporary);
    written
}

fn candidate_record(
    config: MatMulKernelConfig,
    shape: [usize; 3],
) -> Result<CandidateRecord, TunerError> {
    let [m, k, n] = shape;
    if [m, k, n].contains(&0) {
        return Err(TunerError::InvalidArgument("M, K, and N must be nonzero"));
    }
    let [bm, bn, bk] = config.tile();
    let padded_m = m.div_ceil(bm as usize) * bm as usize;
    let padded_n = n.div_ceil(bn as usize) * bn as usize;
    let padded_k = k.div_ceil(bk as usize) * bk as usize;
    let output_ratio = checked_ratio(padded_m.checked_mul(padded_n), m.checked_mul(n))?;
    let reduction_ratio = padded_k as f64 / k as f64;
    let reuse_proxy = (bm as f64 * bn as f64) / (bm + bn) as f64;
    let shared = config.shared_memory_bytes()?;
    let static_score = output_ratio * reduction_ratio
        + f64::from(shared) / 65_536.0
        + 1.0 / reuse_proxy.max(f64::EPSILON);
    let dispatch = [
        u32::try_from(n.div_ceil(bn as usize))
            .map_err(|_| TunerError::InvalidArgument("dispatch exceeds u32"))?,
        u32::try_from(m.div_ceil(bm as usize))
            .map_err(|_| TunerError::InvalidArgument("dispatch exceeds u32"))?,
        1,
    ];
    let id = config.candidate_id();
    Ok(CandidateRecord {
        id: candidate_id(config),
        variant: "tiled".into(),
        tile: id.tile,
        workgroup: id.workgroup,
        shader_variant: id.shader_variant,
        shared_memory_bytes: shared,
        dispatch,
        padded_output_ratio: output_ratio,
        padded_reduction_ratio: reduction_ratio,
        reuse_proxy,
        static_score,
    })
}

fn candidate_id(config: MatMulKernelConfig) -> String {
    let KernelCandidateId {
        tile: [bm, bn, bk],
        workgroup: [x, y, z],
        shader_variant,
        ..
    } = config.candidate_id();
    format!("tiled-bm{bm}-bn{bn}-bk{bk}-wg{x}x{y}x{z}-shader{shader_variant}")
}

fn checked_ratio(numerator: Option<usize>, denominator: Option<usize>) -> Result<f64, TunerError> {
    let numerator = numerator.ok_or(TunerError::InvalidArgument("padding estimate overflows"))?;
    let denominator = denominator.ok_or(TunerError::InvalidArgument("shape size overflows"))?;
    Ok(numerator as f64 / denominator as f64)
}

fn record_batch<F>(
    candidate: &CandidateRecord,
    history: &mut Vec<SearchDecision>,
    scores: &mut Vec<(usize, Vec<f64>)>,
    candidate_index: usize,
    reason: &str,
    measure: &mut F,
) where
    F: FnMut(&CandidateRecord) -> Result<BatchObservation, String>,
{
    let batch_index = history.len();
    match measure(candidate) {
        Ok(observation)
            if observation.correctness_passed
                && !observation.samples_ns.is_empty()
                && observation
                    .samples_ns
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0) =>
        {
            scores
                .iter_mut()
                .find(|(index, _)| *index == candidate_index)
                .map(|(_, samples)| samples.extend_from_slice(&observation.samples_ns))
                .unwrap_or_else(|| scores.push((candidate_index, observation.samples_ns.clone())));
            history.push(SearchDecision {
                batch_index,
                candidate_id: candidate.id.clone(),
                reason: reason.into(),
                median_ns: Statistics::calculate(&observation.samples_ns)
                    .ok()
                    .map(|stats| stats.median),
                raw_samples_ns: observation.samples_ns,
                correctness_passed: true,
                error: None,
            });
        }
        Ok(observation) => history.push(SearchDecision {
            batch_index,
            candidate_id: candidate.id.clone(),
            reason: reason.into(),
            raw_samples_ns: observation.samples_ns,
            median_ns: None,
            correctness_passed: observation.correctness_passed,
            error: Some("invalid timing samples or failed correctness".into()),
        }),
        Err(error) => history.push(SearchDecision {
            batch_index,
            candidate_id: candidate.id.clone(),
            reason: reason.into(),
            raw_samples_ns: Vec::new(),
            median_ns: None,
            correctness_passed: false,
            error: Some(error),
        }),
    }
}

fn best_score(scores: &[(usize, Vec<f64>)]) -> Option<(usize, f64)> {
    scores
        .iter()
        .filter_map(|(index, samples)| {
            Statistics::calculate(samples)
                .ok()
                .map(|stats| (*index, stats.median))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

fn unmeasured_outcome(
    policy: Policy,
    selected: &CandidateRecord,
    considered: usize,
    reason: &str,
    decision_overhead_ns: u64,
) -> SearchOutcome {
    SearchOutcome {
        policy,
        selected_candidate_id: selected.id.clone(),
        selected_score_ns: None,
        candidates_considered: considered,
        candidates_measured: 0,
        measurement_batches: 0,
        total_measured_dispatches: 0,
        search_decision_overhead_ns: decision_overhead_ns,
        ordered_history: Vec::new(),
        termination_reason: reason.into(),
    }
}

fn seeded_shuffle(values: &mut [usize], mut state: u64) {
    for index in (1..values.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.swap(index, (state as usize) % (index + 1));
    }
}

fn correctness(id: &str, actual: &[f32], expected: &[f32]) -> CorrectnessAdmission {
    let mut maximum_absolute_error = 0.0f32;
    let mut maximum_relative_error = 0.0f32;
    let mut mismatch_count = actual.len().abs_diff(expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(ABSOLUTE_TOLERANCE);
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        if !actual.is_finite()
            || !expected.is_finite()
            || !within_tolerance(actual, expected, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
        {
            mismatch_count += 1;
        }
    }
    CorrectnessAdmission {
        candidate_id: id.into(),
        passed: mismatch_count == 0,
        maximum_absolute_error,
        maximum_relative_error,
        mismatch_count,
    }
}

fn sample_value(
    sample: &constraintforge_gpu::PreparedSample,
    mode: MeasurementMode,
) -> Result<f64, &'static str> {
    match mode {
        MeasurementMode::GpuTimestamp => sample
            .gpu_compute_pass_duration_ns
            .ok_or("GPU timestamp missing"),
        MeasurementMode::HostPreparedEndToEnd => Ok(sample.host_duration_ns as f64),
    }
}

fn adapter_record(
    adapter: &AdapterSummary,
    timestamp_period_ns: Option<f32>,
) -> AdapterSummaryRecord {
    AdapterSummaryRecord {
        name: adapter.name.clone(),
        backend: adapter.backend.clone(),
        device_type: adapter.device_type.clone(),
        max_buffer_size: adapter.max_buffer_size,
        max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: adapter.max_compute_workgroups_per_dimension,
        timestamp_query_supported: timestamp_period_ns.is_some(),
        timestamp_period_ns,
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn unix_seconds() -> Result<u64, TunerError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| TunerError::InvalidArgument("system clock precedes Unix epoch"))?
        .as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records() -> Vec<CandidateRecord> {
        candidate_records([31, 33, 29]).unwrap()
    }

    fn budget(candidates: usize, batches: usize) -> SearchBudget {
        SearchBudget {
            max_candidates: candidates,
            max_batches: batches,
            max_duration: Duration::from_secs(1),
        }
    }

    #[test]
    fn candidates_are_unique_deterministic_and_real() {
        let first = records();
        let second = records();
        assert_eq!(
            first.iter().map(|value| &value.id).collect::<Vec<_>>(),
            second.iter().map(|value| &value.id).collect::<Vec<_>>()
        );
        assert_eq!(first.len(), 10);
        assert_eq!(
            first
                .iter()
                .map(|value| &value.id)
                .collect::<HashSet<_>>()
                .len(),
            10
        );
        assert!(first.iter().any(|value| value.tile == [16, 16, 32]));
        assert!(first.iter().all(|value| value.shared_memory_bytes > 0));
        assert!(
            DEVELOPMENT_SHAPES
                .iter()
                .all(|shape| !HELD_OUT_SHAPES.contains(shape))
        );
    }

    #[test]
    fn fixed_and_heuristic_use_no_measurements() {
        for policy in [Policy::Fixed, Policy::Heuristic] {
            let mut calls = 0;
            let result = schedule(&records(), policy, budget(0, 0), 42, |_| {
                calls += 1;
                unreachable!()
            })
            .unwrap();
            assert_eq!(calls, 0);
            assert_eq!(result.measurement_batches, 0);
            assert_eq!(result.total_measured_dispatches, 0);
        }
    }

    #[test]
    fn random_is_seeded_unique_and_budgeted() {
        let run = || {
            schedule(&records(), Policy::Random, budget(4, 8), 42, |candidate| {
                Ok(BatchObservation {
                    samples_ns: vec![candidate.static_score],
                    correctness_passed: true,
                })
            })
            .unwrap()
        };
        let first = run();
        let second = run();
        assert_eq!(
            first
                .ordered_history
                .iter()
                .map(|value| &value.candidate_id)
                .collect::<Vec<_>>(),
            second
                .ordered_history
                .iter()
                .map(|value| &value.candidate_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(first.measurement_batches, 4);
        assert_eq!(first.candidates_measured, 4);
        assert_eq!(first.total_measured_dispatches, 4);
    }

    #[test]
    fn matched_batch_design_equalizes_budgeted_policies() {
        for candidates in [2, 4, 6, 8] {
            for policy in [
                Policy::Random,
                Policy::ResourceRankedOnly,
                Policy::ResourceAware,
            ] {
                let run = || {
                    schedule_with_design(
                        &records(),
                        policy,
                        budget(candidates, 8),
                        42,
                        SearchDesign::MatchedBatchesV2,
                        |candidate| {
                            Ok(BatchObservation {
                                samples_ns: vec![
                                    candidate.static_score,
                                    candidate.static_score + 1.0,
                                    candidate.static_score + 2.0,
                                ],
                                correctness_passed: true,
                            })
                        },
                    )
                    .unwrap()
                };
                let first = run();
                let second = run();
                assert_eq!(first.measurement_batches, 8);
                assert_eq!(first.total_measured_dispatches, 24);
                assert!(first.candidates_measured <= candidates);
                assert_eq!(first.selected_candidate_id, second.selected_candidate_id);
                assert_eq!(first.ordered_history, second.ordered_history);
                assert!(
                    records()
                        .iter()
                        .any(|candidate| candidate.id == first.selected_candidate_id)
                );
            }
        }
    }

    #[test]
    fn matched_random_repeats_exhausted_pool_and_records_failures() {
        let pool = &records()[..1];
        let result = schedule_with_design(
            pool,
            Policy::Random,
            budget(8, 8),
            9,
            SearchDesign::MatchedBatchesV2,
            |_| {
                Ok(BatchObservation {
                    samples_ns: vec![1.0, 2.0, 3.0],
                    correctness_passed: true,
                })
            },
        )
        .unwrap();
        assert_eq!(result.candidates_measured, 1);
        assert_eq!(result.measurement_batches, 8);
        assert_eq!(result.total_measured_dispatches, 24);

        let mut attempts = 0;
        assert!(matches!(
            schedule_with_design(
                pool,
                Policy::Random,
                budget(1, 8),
                9,
                SearchDesign::MatchedBatchesV2,
                |_| {
                    attempts += 1;
                    Err("injected measurement failure".into())
                }
            ),
            Err(TunerError::NoValidCandidate)
        ));
        assert_eq!(attempts, 8);
    }

    #[test]
    fn matched_design_ties_are_deterministic_and_oversized_budgets_fail() {
        let run = || {
            schedule_with_design(
                &records(),
                Policy::Random,
                budget(4, 8),
                42,
                SearchDesign::MatchedBatchesV2,
                |_| {
                    Ok(BatchObservation {
                        samples_ns: vec![1.0, 1.0, 1.0],
                        correctness_passed: true,
                    })
                },
            )
            .unwrap()
        };
        assert_eq!(run().selected_candidate_id, run().selected_candidate_id);
        assert!(
            SearchBudget {
                max_candidates: usize::MAX,
                max_batches: usize::MAX,
                max_duration: Duration::from_secs(1),
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn resource_aware_accounts_for_repeat_batches() {
        let result = schedule(
            &records(),
            Policy::ResourceAware,
            budget(3, 5),
            7,
            |candidate| {
                Ok(BatchObservation {
                    samples_ns: vec![candidate.static_score, candidate.static_score + 1.0],
                    correctness_passed: true,
                })
            },
        )
        .unwrap();
        assert_eq!(result.candidates_measured, 3);
        assert_eq!(result.measurement_batches, 5);
        assert_eq!(result.total_measured_dispatches, 10);
        assert_eq!(result.ordered_history.len(), 5);
    }

    #[test]
    fn resource_ranked_only_measures_distinct_candidates_without_adaptation() {
        let result = schedule(
            &records(),
            Policy::ResourceRankedOnly,
            budget(3, 8),
            7,
            |candidate| {
                Ok(BatchObservation {
                    samples_ns: vec![candidate.static_score],
                    correctness_passed: true,
                })
            },
        )
        .unwrap();
        assert_eq!(result.candidates_measured, 3);
        assert_eq!(result.measurement_batches, 3);
        assert_eq!(
            result
                .ordered_history
                .iter()
                .map(|value| &value.candidate_id)
                .collect::<HashSet<_>>()
                .len(),
            3
        );
    }

    #[test]
    fn zero_time_and_failures_terminate_without_fake_selection() {
        assert!(matches!(
            schedule(
                &records(),
                Policy::Random,
                SearchBudget {
                    max_candidates: 2,
                    max_batches: 2,
                    max_duration: Duration::ZERO
                },
                1,
                |_| unreachable!()
            ),
            Err(TunerError::NoValidCandidate)
        ));
        assert!(matches!(
            schedule(&records(), Policy::Random, budget(2, 2), 1, |_| Err(
                "compile failure".into()
            )),
            Err(TunerError::NoValidCandidate)
        ));
        assert!(matches!(
            schedule(&records(), Policy::Random, budget(2, 2), 1, |_| Ok(
                BatchObservation {
                    samples_ns: vec![f64::NAN],
                    correctness_passed: true
                }
            )),
            Err(TunerError::NoValidCandidate)
        ));
    }

    #[test]
    fn result_schema_serializes_and_output_does_not_overwrite() {
        let outcome = schedule(
            &records(),
            Policy::Fixed,
            budget(0, 0),
            1,
            |_| unreachable!(),
        )
        .unwrap();
        let result = TuningResult {
            schema_version: TUNING_SCHEMA_VERSION.into(),
            trial_id: "test".into(),
            utc_unix_seconds: 0,
            shape_mkn: [1, 1, 1],
            input_seed: 1,
            search_seed: 1,
            policy_version: POLICY_VERSION.into(),
            search_design: None,
            measurement_mode: MeasurementMode::GpuTimestamp,
            budget_max_candidates: 0,
            budget_max_batches: 0,
            budget_max_duration_ms: 0,
            adapter: AdapterSummaryRecord {
                name: "test".into(),
                backend: "test".into(),
                device_type: "test".into(),
                max_buffer_size: 1,
                max_storage_buffer_binding_size: 1,
                max_compute_workgroups_per_dimension: 1,
                timestamp_query_supported: true,
                timestamp_period_ns: Some(1.0),
            },
            software: current_software_metadata().unwrap(),
            candidate_pool: records(),
            static_rejections: vec![],
            correctness_admission: vec![],
            search: outcome,
            independent_evaluation_samples_ns: vec![1.0],
            independent_evaluation_statistics_ns: Statistics::calculate(&[1.0]).unwrap(),
            preparation_time_ns: 0,
            correctness_validation_time_ns: 0,
            search_wall_time_ns: 0,
            tuning_wall_time_ns: 0,
            search_decision_overhead_ns: 0,
            compiled_candidates: 0,
            complete: true,
            errors: vec![],
        };
        assert!(
            serde_json::to_string(&result)
                .unwrap()
                .contains(TUNING_SCHEMA_VERSION)
        );
        let path = std::env::temp_dir().join(format!(
            "constraintforge-tuner-test-{}.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        write_result(&path, &result).unwrap();
        assert!(matches!(
            write_result(&path, &result),
            Err(TunerError::OutputExists(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires a real GPU adapter with timestamp queries"]
    fn real_gpu_small_budget_random_and_resource_aware_tuning() {
        for policy in [Policy::Random, Policy::ResourceAware] {
            let result = run_gpu(&GpuTuningConfig {
                shape: [31, 33, 29],
                workload_seed: 42,
                search_seed: 42,
                policy,
                score_ablation: ScoreAblation::Full,
                search_design: SearchDesign::Legacy,
                budget: SearchBudget {
                    max_candidates: 3,
                    max_batches: 4,
                    max_duration: Duration::from_secs(30),
                },
                measurement_mode: MeasurementMode::GpuTimestamp,
                tuning_warmups: 1,
                tuning_runs: 2,
                evaluation_warmups: 1,
                evaluation_runs: 3,
            })
            .unwrap();
            assert!(result.complete);
            assert!(result.search.measurement_batches <= 4);
            assert!(result.search.candidates_measured <= 3);
            assert_eq!(result.independent_evaluation_samples_ns.len(), 3);
            assert!(
                result
                    .correctness_admission
                    .iter()
                    .all(|value| value.passed)
            );
        }
    }
}

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

use constraintforge_bench::{SoftwareMetadata, current_software_metadata};
use constraintforge_tuner::{
    AdapterSummaryRecord, GpuReferenceBatchConfig, GpuTuningConfig, MeasurementMode, Policy,
    ReferenceBlockResult, ScoreAblation, SearchBudget, SearchDesign, TUNING_SCHEMA_VERSION,
    TuningResult, run_gpu, run_gpu_reference_batch, tiled_candidate_ids,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ResearchError, atomic_json, now, write_new};

pub const PROSPECTIVE_PROTOCOL_SCHEMA: &str = "constraintforge-m7-prospective-protocol-v1";
pub const PROSPECTIVE_SESSION_SCHEMA: &str = "constraintforge-m7-prospective-session-v2";
pub const REFERENCE_RECORD_SCHEMA: &str = "constraintforge-m7-candidate-reference-v2";
pub const CORRECTED_IMPLEMENTATION_VERSION: &str = "m7-phase2a-v1-corrected-1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProspectiveProtocol {
    pub schema: String,
    pub id: String,
    pub version: String,
    pub implementation_version: String,
    pub phase: String,
    pub frozen: bool,
    pub research_question: String,
    pub hardware_target: String,
    pub candidate_space: String,
    pub measurement: String,
    pub ordering: String,
    pub primary_estimand: String,
    pub reference_aggregation: String,
    pub relative_regret: String,
    pub tie_rule: String,
    pub invalid_rule: String,
    pub missing_rule: String,
    pub selection_mismatch_rule: String,
    pub dependence: String,
    pub stop_rules: String,
    pub workloads: Vec<[usize; 3]>,
    pub budgets: Vec<usize>,
    pub policies: Vec<String>,
    pub seeds: Vec<u64>,
    pub sessions: usize,
    pub max_batches: usize,
    pub max_duration_ms: u64,
    pub tuning_warmups: u32,
    pub tuning_runs: u32,
    pub evaluation_warmups: u32,
    pub evaluation_runs: u32,
    pub reference_blocks: usize,
    pub reference_warmups_per_block: u32,
    pub reference_runs_per_block: u32,
    pub reference_max_duration_ms: u64,
    pub session_wall_cap_ms: u64,
    pub expected_search_trials: usize,
    pub expected_reference_records: usize,
    pub expected_reference_blocks: usize,
    pub expected_timed_search_samples: usize,
    pub expected_selected_evaluation_samples: usize,
    pub expected_reference_samples: usize,
    pub expected_total_timed_samples: usize,
    pub expected_total_candidate_invocations: usize,
    pub expected_raw_artifacts: usize,
}

impl ProspectiveProtocol {
    pub fn parse(path: &Path) -> Result<Self, ResearchError> {
        let protocol: Self = toml::from_str(&fs::read_to_string(path)?)?;
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ResearchError> {
        let trials =
            self.sessions * self.workloads.len() * self.budgets.len() * self.policies.len();
        let candidates = tiled_candidate_ids().len();
        let references = self.sessions * self.workloads.len() * candidates;
        let reference_blocks = references * self.reference_blocks;
        let timed_search = trials * self.max_batches * self.tuning_runs as usize;
        let selected = trials * self.evaluation_runs as usize;
        let reference_samples = reference_blocks * self.reference_runs_per_block as usize;
        let correctness = trials * candidates;
        let search_warmups = trials * self.max_batches * self.tuning_warmups as usize;
        let selected_warmups = trials * self.evaluation_warmups as usize;
        let reference_warmups = reference_blocks * self.reference_warmups_per_block as usize;
        let invocations = timed_search
            + selected
            + reference_samples
            + correctness
            + search_warmups
            + selected_warmups
            + reference_warmups;
        if self.schema != PROSPECTIVE_PROTOCOL_SCHEMA
            || self.implementation_version != "m7-phase2a-v1"
            || self.phase != "frozen"
            || !self.frozen
            || self.workloads != [[7, 19, 11], [256, 64, 384]]
            || self.budgets != [2, 8]
            || self.policies != ["random", "resource-aware"]
            || self.sessions != 5
            || self.seeds != [71, 83, 97, 109, 127]
            || candidates != 10
            || self.max_batches != 8
            || self.tuning_runs != 3
            || self.evaluation_runs != 10
            || self.reference_blocks != 3
            || self.reference_runs_per_block != 10
            || self.max_duration_ms != 60_000
            || self.reference_max_duration_ms != 60_000
            || self.expected_search_trials != trials
            || self.expected_reference_records != references
            || self.expected_reference_blocks != reference_blocks
            || self.expected_timed_search_samples != timed_search
            || self.expected_selected_evaluation_samples != selected
            || self.expected_reference_samples != reference_samples
            || self.expected_total_timed_samples != timed_search + selected + reference_samples
            || self.expected_total_candidate_invocations != invocations
            || self.expected_raw_artifacts != self.sessions + trials + references + 1
        {
            return Err(ResearchError::Invalid(
                "M7 Phase 2A protocol differs from the approved bounded pilot",
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ResearchError> {
        Ok(format!(
            "{:x}",
            Sha256::digest(toml::to_string_pretty(self)?)
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedSearchTrial {
    pub trial_id: String,
    pub workload: [usize; 3],
    pub budget: usize,
    pub policy_id: String,
    pub seed: u64,
    pub order_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedReferenceBlock {
    pub workload: [usize; 3],
    pub block_index: usize,
    pub candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProspectivePlan {
    pub schema: String,
    pub session_index: usize,
    pub seed: u64,
    pub search_trials: Vec<PlannedSearchTrial>,
    pub reference_blocks: Vec<PlannedReferenceBlock>,
}

pub fn prospective_plan(
    protocol: &ProspectiveProtocol,
    session_index: usize,
) -> Result<ProspectivePlan, ResearchError> {
    protocol.validate()?;
    if session_index >= protocol.sessions {
        return Err(ResearchError::Invalid("session index exceeds protocol"));
    }
    let seed = protocol.seeds[session_index];
    let mut workloads = protocol.workloads.clone();
    if session_index % 2 == 1 {
        workloads.reverse();
    }
    let mut search_trials = Vec::new();
    for (wi, workload) in workloads.iter().enumerate() {
        let mut budgets = protocol.budgets.clone();
        if (session_index + wi) % 2 == 1 {
            budgets.reverse();
        }
        for (bi, budget) in budgets.iter().enumerate() {
            let mut policies = protocol.policies.clone();
            if (session_index + wi + bi) % 2 == 1 {
                policies.reverse();
            }
            for policy in policies {
                let order_index = search_trials.len();
                search_trials.push(PlannedSearchTrial {
                    trial_id: format!(
                        "s{:02}-m{}k{}n{}-b{}-{}-seed{}",
                        session_index + 1,
                        workload[0],
                        workload[1],
                        workload[2],
                        budget,
                        policy,
                        seed
                    ),
                    workload: *workload,
                    budget: *budget,
                    policy_id: policy,
                    seed,
                    order_index,
                });
            }
        }
    }
    let candidates = tiled_candidate_ids();
    let mut reference_blocks = Vec::new();
    for (wi, workload) in workloads.iter().enumerate() {
        for block_index in 0..protocol.reference_blocks {
            let mut order = candidates.clone();
            let shift = (session_index + wi * 2 + block_index * 3) % order.len();
            order.rotate_left(shift);
            if (session_index + wi + block_index) % 2 == 1 {
                order.reverse();
            }
            reference_blocks.push(PlannedReferenceBlock {
                workload: *workload,
                block_index,
                candidate_ids: order,
            });
        }
    }
    Ok(ProspectivePlan {
        schema: "constraintforge-m7-prospective-plan-v1".into(),
        session_index,
        seed,
        search_trials,
        reference_blocks,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentObservation {
    pub captured_utc: u64,
    pub power_source: Option<String>,
    pub battery_percent: Option<String>,
    pub thermal_state: Option<String>,
    pub background_load: Option<String>,
}

fn environment_observation() -> Result<EnvironmentObservation, ResearchError> {
    Ok(EnvironmentObservation {
        captured_utc: now()?,
        power_source: std::env::var("CONSTRAINTFORGE_POWER_SOURCE").ok(),
        battery_percent: std::env::var("CONSTRAINTFORGE_BATTERY_PERCENT").ok(),
        thermal_state: std::env::var("CONSTRAINTFORGE_THERMAL_STATE").ok(),
        background_load: std::env::var("CONSTRAINTFORGE_BACKGROUND_LOAD").ok(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProspectiveSearchRecord {
    pub plan: PlannedSearchTrial,
    pub status: String,
    pub result_file: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateReferenceRecord {
    pub schema: String,
    pub session_id: String,
    pub session_index: usize,
    pub protocol_sha256: String,
    pub source_revision: String,
    pub source_tree_sha256: Option<String>,
    pub workload: [usize; 3],
    pub candidate_id: String,
    pub correctness_evidence_trial_ids: Vec<String>,
    pub selection_seal_sha256: String,
    pub adapter: Option<AdapterSummaryRecord>,
    pub blocks: Vec<ReferenceBlockResult>,
    pub preparation_time_ns: u64,
    pub measurement_wall_time_ns: u64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceAttemptRecord {
    pub workload: [usize; 3],
    pub candidate_id: String,
    pub block_index: usize,
    pub status: String,
    pub started_utc: u64,
    pub ended_utc: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProspectiveSessionRecord {
    pub schema: String,
    pub session_id: String,
    pub session_index: usize,
    pub protocol_sha256: String,
    pub software: SoftwareMetadata,
    pub plan: ProspectivePlan,
    pub phase: String,
    pub search: Vec<ProspectiveSearchRecord>,
    pub selections_sealed: bool,
    pub selection_seal_sha256: Option<String>,
    pub reference_files: Vec<String>,
    #[serde(default)]
    pub reference_attempts: Vec<ReferenceAttemptRecord>,
    pub environment: Vec<EnvironmentObservation>,
    #[serde(default)]
    pub source_preflight_captured_utc: u64,
    #[serde(default)]
    pub generated_output_created_after_preflight: bool,
    pub started_utc: u64,
    pub ended_utc: Option<u64>,
    pub interruptions: Vec<String>,
}

pub fn run_prospective_session(
    protocol: &ProspectiveProtocol,
    session_id: &str,
    session_index: usize,
    output: &Path,
    resume: bool,
) -> Result<ProspectiveSessionRecord, ResearchError> {
    run_prospective_session_with(
        ProspectiveRunRequest {
            protocol,
            session_id,
            session_index,
            output,
            resume,
        },
        || current_software_metadata().map_err(ResearchError::from),
        |config| run_gpu(config).map_err(|error| error.to_string()),
        |config| run_gpu_reference_batch(config).map_err(|error| error.to_string()),
    )
}

type ReferenceRunResult = (
    AdapterSummaryRecord,
    SoftwareMetadata,
    ReferenceBlockResult,
    u64,
    u64,
);

struct ProspectiveRunRequest<'a> {
    protocol: &'a ProspectiveProtocol,
    session_id: &'a str,
    session_index: usize,
    output: &'a Path,
    resume: bool,
}

fn run_prospective_session_with<Metadata, Search, Reference>(
    request: ProspectiveRunRequest<'_>,
    mut metadata: Metadata,
    mut search_gpu: Search,
    mut reference_gpu: Reference,
) -> Result<ProspectiveSessionRecord, ResearchError>
where
    Metadata: FnMut() -> Result<SoftwareMetadata, ResearchError>,
    Search: FnMut(&GpuTuningConfig) -> Result<TuningResult, String>,
    Reference: FnMut(&GpuReferenceBatchConfig) -> Result<ReferenceRunResult, String>,
{
    let ProspectiveRunRequest {
        protocol,
        session_id,
        session_index,
        output,
        resume,
    } = request;
    protocol.validate()?;
    validate_output_path(output)?;
    let plan = prospective_plan(protocol, session_index)?;
    let protocol_sha256 = protocol.sha256()?;
    let state_path = output.join("session.json");
    let protocol_path = output.join("resolved-protocol.toml");
    if output.exists() && !resume {
        return Err(ResearchError::OutputExists(output.into()));
    }
    if resume && !output.exists() {
        return Err(ResearchError::Invalid(
            "cannot resume a prospective session that does not exist",
        ));
    }
    let source_preflight_captured_utc = now()?;
    let mut current_software = metadata()?;
    current_software.research_implementation_version =
        Some(CORRECTED_IMPLEMENTATION_VERSION.into());
    fs::create_dir_all(output)?;
    if !protocol_path.exists() {
        write_new(&protocol_path, toml::to_string_pretty(protocol)?.as_bytes())?;
    }
    let mut state = if state_path.exists() {
        let state: ProspectiveSessionRecord = serde_json::from_slice(&fs::read(&state_path)?)?;
        let saved = ProspectiveProtocol::parse(&protocol_path)?;
        if !resume
            || state.session_id != session_id
            || state.session_index != session_index
            || state.protocol_sha256 != protocol_sha256
            || saved.sha256()? != protocol_sha256
            || state.software.source_revision != current_software.source_revision
            || state.software.source_tree_sha256 != current_software.source_tree_sha256
            || state.plan != plan
        {
            return Err(ResearchError::Invalid(
                "prospective resume protocol, plan, session, or source mismatch",
            ));
        }
        state
    } else {
        ProspectiveSessionRecord {
            schema: PROSPECTIVE_SESSION_SCHEMA.into(),
            session_id: session_id.into(),
            session_index,
            protocol_sha256: protocol_sha256.clone(),
            software: current_software,
            plan: plan.clone(),
            phase: "search".into(),
            search: Vec::new(),
            selections_sealed: false,
            selection_seal_sha256: None,
            reference_files: Vec::new(),
            reference_attempts: Vec::new(),
            environment: vec![environment_observation()?],
            source_preflight_captured_utc,
            generated_output_created_after_preflight: true,
            started_utc: now()?,
            ended_utc: None,
            interruptions: Vec::new(),
        }
    };
    atomic_json(&state_path, &state)?;
    reconcile_search_orphans(output, &mut state)?;
    atomic_json(&state_path, &state)?;
    let session_started = Instant::now();
    for trial in &plan.search_trials {
        if state
            .search
            .iter()
            .any(|record| record.plan.trial_id == trial.trial_id)
        {
            continue;
        }
        if session_started.elapsed() > Duration::from_millis(protocol.session_wall_cap_ms) {
            interrupt(&state_path, &mut state, "session wall cap reached")?;
            return Err(ResearchError::Invalid(
                "prospective session wall cap reached",
            ));
        }
        let policy = match trial.policy_id.as_str() {
            "random" => Policy::Random,
            "resource-aware" => Policy::ResourceAware,
            _ => return Err(ResearchError::Invalid("unknown prospective policy")),
        };
        let run = search_gpu(&GpuTuningConfig {
            shape: trial.workload,
            workload_seed: trial.seed,
            search_seed: trial.seed,
            policy,
            score_ablation: ScoreAblation::Full,
            search_design: SearchDesign::MatchedBatchesV2,
            budget: SearchBudget {
                max_candidates: trial.budget,
                max_batches: protocol.max_batches,
                max_duration: Duration::from_millis(protocol.max_duration_ms),
            },
            measurement_mode: MeasurementMode::GpuTimestamp,
            tuning_warmups: protocol.tuning_warmups,
            tuning_runs: protocol.tuning_runs,
            evaluation_warmups: protocol.evaluation_warmups,
            evaluation_runs: protocol.evaluation_runs,
        });
        let (status, result_file, error) = match run {
            Ok(mut result) => {
                result.trial_id = trial.trial_id.clone();
                result.software = state.software.clone();
                validate_search_result(protocol, trial, &result)?;
                let relative = format!("trials/{}.json", trial.trial_id);
                write_new(
                    &output.join(&relative),
                    &serde_json::to_vec_pretty(&result)?,
                )?;
                ("complete".into(), Some(relative), None)
            }
            Err(error) => ("failed".into(), None, Some(error.to_string())),
        };
        state.search.push(ProspectiveSearchRecord {
            plan: trial.clone(),
            status,
            result_file,
            error,
        });
        atomic_json(&state_path, &state)?;
    }
    if state
        .search
        .iter()
        .any(|record| record.status != "complete")
    {
        interrupt(
            &state_path,
            &mut state,
            "search failure prevents selection sealing",
        )?;
        return Ok(state);
    }
    if !state.selections_sealed {
        let seal = selection_seal(output, &state, protocol)?;
        state.selection_seal_sha256 = Some(seal);
        state.selections_sealed = true;
        state.phase = "reference".into();
        atomic_json(&state_path, &state)?;
    }
    let seal = state
        .selection_seal_sha256
        .clone()
        .ok_or(ResearchError::Invalid(
            "reference phase lacks selection seal",
        ))?;
    initialize_reference_storage(protocol, output, &state)?;
    reconcile_reference_records(protocol, output, &mut state)?;
    reconcile_reference_attempts(output, &mut state)?;
    atomic_json(&state_path, &state)?;
    for reference_block in &plan.reference_blocks {
        for (order_index, candidate_id) in reference_block.candidate_ids.iter().enumerate() {
            let relative = reference_path(reference_block.workload, candidate_id);
            let path = output.join(&relative);
            let mut record = if path.exists() {
                let record: CandidateReferenceRecord = serde_json::from_slice(&fs::read(&path)?)?;
                validate_reference_identity(
                    &record,
                    &state,
                    reference_block.workload,
                    candidate_id,
                )?;
                record
            } else {
                CandidateReferenceRecord {
                    schema: REFERENCE_RECORD_SCHEMA.into(),
                    session_id: state.session_id.clone(),
                    session_index,
                    protocol_sha256: protocol_sha256.clone(),
                    source_revision: state.software.source_revision.clone(),
                    source_tree_sha256: state.software.source_tree_sha256.clone(),
                    workload: reference_block.workload,
                    candidate_id: candidate_id.clone(),
                    correctness_evidence_trial_ids: correctness_evidence(
                        &state,
                        reference_block.workload,
                    )?,
                    selection_seal_sha256: seal.clone(),
                    adapter: None,
                    blocks: Vec::new(),
                    preparation_time_ns: 0,
                    measurement_wall_time_ns: 0,
                    status: "incomplete".into(),
                    error: None,
                }
            };
            if record.status == "complete"
                || record
                    .blocks
                    .iter()
                    .any(|block| block.block_index == reference_block.block_index)
            {
                continue;
            }
            if record.status == "failed" {
                return Err(ResearchError::Invalid(
                    "failed reference record cannot be resumed or remeasured",
                ));
            }
            if session_started.elapsed() > Duration::from_millis(protocol.session_wall_cap_ms) {
                interrupt(&state_path, &mut state, "session wall cap reached")?;
                return Err(ResearchError::Invalid(
                    "prospective session wall cap reached",
                ));
            }
            begin_reference_attempt(
                &state_path,
                &mut state,
                reference_block.workload,
                candidate_id,
                reference_block.block_index,
            )?;
            let run = reference_gpu(&GpuReferenceBatchConfig {
                shape: reference_block.workload,
                workload_seed: plan.seed,
                candidate_id: candidate_id.clone(),
                block_index: reference_block.block_index,
                order_index,
                warmups: protocol.reference_warmups_per_block,
                samples: protocol.reference_runs_per_block,
            });
            let (adapter, software, block, preparation_time_ns, measurement_wall_time_ns) =
                match run {
                    Ok(result) => result,
                    Err(error) => {
                        record.status = "failed".into();
                        record.error = Some(error.clone());
                        atomic_json(&path, &record)?;
                        finish_reference_attempt(
                            &state_path,
                            &mut state,
                            reference_block.workload,
                            candidate_id,
                            reference_block.block_index,
                            "failed_with_durable_evidence",
                            Some(error),
                        )?;
                        interrupt(&state_path, &mut state, "reference batch failed")?;
                        return Ok(state);
                    }
                };
            if software.source_revision != state.software.source_revision
                || software.source_tree_sha256 != state.software.source_tree_sha256
            {
                return Err(ResearchError::Invalid(
                    "reference source identity changed after selection seal",
                ));
            }
            if record
                .adapter
                .as_ref()
                .is_some_and(|saved| !same_adapter(saved, &adapter))
            {
                return Err(ResearchError::Invalid(
                    "reference adapter changed within candidate record",
                ));
            }
            record.adapter = Some(adapter);
            record.preparation_time_ns = record
                .preparation_time_ns
                .saturating_add(preparation_time_ns);
            record.measurement_wall_time_ns = record
                .measurement_wall_time_ns
                .saturating_add(measurement_wall_time_ns);
            if record.measurement_wall_time_ns
                > protocol.reference_max_duration_ms.saturating_mul(1_000_000)
            {
                record.status = "failed".into();
                record.error = Some("candidate reference wall cap exceeded".into());
                atomic_json(&path, &record)?;
                finish_reference_attempt(
                    &state_path,
                    &mut state,
                    reference_block.workload,
                    candidate_id,
                    reference_block.block_index,
                    "failed_with_durable_evidence",
                    Some("candidate reference wall cap exceeded".into()),
                )?;
                interrupt(
                    &state_path,
                    &mut state,
                    "candidate reference wall cap exceeded",
                )?;
                return Ok(state);
            }
            record.blocks.push(block);
            record.blocks.sort_by_key(|block| block.block_index);
            if record.blocks.len() == protocol.reference_blocks {
                validate_complete_reference(protocol, &record)?;
                record.status = "complete".into();
            }
            atomic_json(&path, &record)?;
            finish_reference_attempt(
                &state_path,
                &mut state,
                reference_block.workload,
                candidate_id,
                reference_block.block_index,
                "completed_and_durably_recorded",
                None,
            )?;
            if record.status == "complete" && !state.reference_files.contains(&relative) {
                state.reference_files.push(relative);
                state.reference_files.sort();
                atomic_json(&state_path, &state)?;
            }
        }
    }
    validate_complete_session(protocol, &state, output)?;
    state.phase = "complete".into();
    state.environment.push(environment_observation()?);
    state.ended_utc = Some(now()?);
    atomic_json(&state_path, &state)?;
    Ok(state)
}

fn reconcile_reference_records(
    protocol: &ProspectiveProtocol,
    output: &Path,
    state: &mut ProspectiveSessionRecord,
) -> Result<(), ResearchError> {
    for workload in &protocol.workloads {
        for candidate in tiled_candidate_ids() {
            let relative = reference_path(*workload, &candidate);
            let path = output.join(&relative);
            if !path.exists() || state.reference_files.contains(&relative) {
                continue;
            }
            let record: CandidateReferenceRecord = serde_json::from_slice(&fs::read(path)?)?;
            validate_reference_identity(&record, state, *workload, &candidate)?;
            if record.status == "complete" {
                validate_complete_reference(protocol, &record)?;
                state.reference_files.push(relative);
                state
                    .interruptions
                    .push("reconciled immutable completed candidate reference record".into());
            } else if record.status == "failed" {
                return Err(ResearchError::Invalid(
                    "failed reference record cannot be resumed or remeasured",
                ));
            }
        }
    }
    state.reference_files.sort();
    Ok(())
}

fn initialize_reference_storage(
    protocol: &ProspectiveProtocol,
    output: &Path,
    state: &ProspectiveSessionRecord,
) -> Result<(), ResearchError> {
    if !output.join("session.json").is_file()
        || !state.selections_sealed
        || selection_seal(output, state, protocol)?
            != state.selection_seal_sha256.clone().unwrap_or_default()
    {
        return Err(ResearchError::Invalid(
            "reference storage requires a durable valid selection seal",
        ));
    }
    let mut parents = BTreeSet::new();
    for workload in &protocol.workloads {
        for candidate in tiled_candidate_ids() {
            let path = output.join(reference_path(*workload, &candidate));
            let parent = path.parent().ok_or(ResearchError::Invalid(
                "candidate reference path has no parent",
            ))?;
            fs::create_dir_all(parent)?;
            parents.insert(parent.to_path_buf());
        }
    }
    for parent in parents {
        let probe = parent.join(format!(".persistence-probe-{}", std::process::id()));
        let result = (|| -> Result<(), std::io::Error> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&probe)?;
            file.write_all(b"reference-persistence-ready")?;
            file.sync_all()
        })();
        let remove_result = probe.exists().then(|| fs::remove_file(&probe)).transpose();
        result?;
        remove_result?;
    }
    Ok(())
}

fn begin_reference_attempt(
    state_path: &Path,
    state: &mut ProspectiveSessionRecord,
    workload: [usize; 3],
    candidate_id: &str,
    block_index: usize,
) -> Result<(), ResearchError> {
    if state.reference_attempts.iter().any(|attempt| {
        attempt.workload == workload
            && attempt.candidate_id == candidate_id
            && attempt.block_index == block_index
    }) {
        return Err(ResearchError::Invalid(
            "reference attempt already exists and cannot be silently remeasured",
        ));
    }
    state.reference_attempts.push(ReferenceAttemptRecord {
        workload,
        candidate_id: candidate_id.into(),
        block_index,
        status: "started_or_possibly_executed".into(),
        started_utc: now()?,
        ended_utc: None,
        error: None,
    });
    atomic_json(state_path, state)
}

fn finish_reference_attempt(
    state_path: &Path,
    state: &mut ProspectiveSessionRecord,
    workload: [usize; 3],
    candidate_id: &str,
    block_index: usize,
    status: &str,
    error: Option<String>,
) -> Result<(), ResearchError> {
    let attempt = state
        .reference_attempts
        .iter_mut()
        .find(|attempt| {
            attempt.workload == workload
                && attempt.candidate_id == candidate_id
                && attempt.block_index == block_index
                && attempt.status == "started_or_possibly_executed"
        })
        .ok_or(ResearchError::Invalid(
            "reference attempt intent is missing",
        ))?;
    attempt.status = status.into();
    attempt.ended_utc = Some(now()?);
    attempt.error = error;
    atomic_json(state_path, state)
}

fn reconcile_reference_attempts(
    output: &Path,
    state: &mut ProspectiveSessionRecord,
) -> Result<(), ResearchError> {
    for attempt in &mut state.reference_attempts {
        match attempt.status.as_str() {
            "completed_and_durably_recorded" => {}
            "failed_with_durable_evidence" => {
                return Err(ResearchError::Invalid(
                    "failed reference attempt cannot be resumed or remeasured",
                ));
            }
            "started_or_possibly_executed" => {
                let path = output.join(reference_path(attempt.workload, &attempt.candidate_id));
                let recorded = path
                    .is_file()
                    .then(|| fs::read(&path))
                    .transpose()?
                    .map(|bytes| serde_json::from_slice::<CandidateReferenceRecord>(&bytes))
                    .transpose()?
                    .is_some_and(|record| {
                        record.blocks.iter().any(|block| {
                            block.block_index == attempt.block_index
                                && block.candidate_id == attempt.candidate_id
                        })
                    });
                if !recorded {
                    return Err(ResearchError::Invalid(
                        "reference attempt may have executed without a durable measurement",
                    ));
                }
                attempt.status = "completed_and_durably_recorded".into();
                attempt.ended_utc = Some(now()?);
                state.interruptions.push(
                    "reconciled reference measurement committed before checkpoint advancement"
                        .into(),
                );
            }
            _ => return Err(ResearchError::Invalid("unknown reference attempt state")),
        }
    }
    Ok(())
}

fn interrupt(
    path: &Path,
    state: &mut ProspectiveSessionRecord,
    reason: &str,
) -> Result<(), ResearchError> {
    state.phase = "interrupted".into();
    state.interruptions.push(reason.into());
    state.environment.push(environment_observation()?);
    state.ended_utc = Some(now()?);
    atomic_json(path, state)
}

fn reconcile_search_orphans(
    output: &Path,
    state: &mut ProspectiveSessionRecord,
) -> Result<(), ResearchError> {
    for trial in &state.plan.search_trials {
        if state
            .search
            .iter()
            .any(|record| record.plan.trial_id == trial.trial_id)
        {
            continue;
        }
        let relative = format!("trials/{}.json", trial.trial_id);
        let path = output.join(&relative);
        if path.exists() {
            let result: TuningResult = serde_json::from_slice(&fs::read(path)?)?;
            if result.trial_id != trial.trial_id
                || result.shape_mkn != trial.workload
                || result.search_seed != trial.seed
                || result.software.source_revision != state.software.source_revision
                || result.software.source_tree_sha256 != state.software.source_tree_sha256
            {
                return Err(ResearchError::Invalid(
                    "orphan prospective trial identity or source mismatch",
                ));
            }
            state.search.push(ProspectiveSearchRecord {
                plan: trial.clone(),
                status: "complete".into(),
                result_file: Some(relative),
                error: None,
            });
            state
                .interruptions
                .push("reconciled immutable prospective search trial".into());
        }
    }
    state.search.sort_by_key(|record| record.plan.order_index);
    Ok(())
}

fn validate_search_result(
    protocol: &ProspectiveProtocol,
    plan: &PlannedSearchTrial,
    result: &TuningResult,
) -> Result<(), ResearchError> {
    let policy = match plan.policy_id.as_str() {
        "random" => Policy::Random,
        "resource-aware" => Policy::ResourceAware,
        _ => return Err(ResearchError::Invalid("unexpected prospective policy")),
    };
    if result.schema_version != TUNING_SCHEMA_VERSION
        || result.shape_mkn != plan.workload
        || result.search.policy != policy
        || result.search.measurement_batches != protocol.max_batches
        || result.search.total_measured_dispatches
            != protocol.max_batches * protocol.tuning_runs as usize
        || result
            .search
            .ordered_history
            .iter()
            .any(|batch| batch.raw_samples_ns.len() != protocol.tuning_runs as usize)
        || result.independent_evaluation_samples_ns.len() != protocol.evaluation_runs as usize
        || result.correctness_admission.len() != 10
        || result
            .correctness_admission
            .iter()
            .any(|admission| !admission.passed)
        || result.budget_max_candidates != plan.budget
        || result.budget_max_batches != protocol.max_batches
        || result.measurement_mode != MeasurementMode::GpuTimestamp
        || !approved_hardware(&result.adapter)
        || !result.complete
    {
        return Err(ResearchError::Invalid(
            "prospective search trial violates the frozen equal-effort design",
        ));
    }
    validate_samples(&result.independent_evaluation_samples_ns)
}

fn selection_seal(
    output: &Path,
    state: &ProspectiveSessionRecord,
    protocol: &ProspectiveProtocol,
) -> Result<String, ResearchError> {
    if state.search.len() != state.plan.search_trials.len() || state.search.len() != 8 {
        return Err(ResearchError::Invalid(
            "all eight selections must exist before reference access",
        ));
    }
    let mut digest = Sha256::new();
    for planned in &state.plan.search_trials {
        let record = state
            .search
            .iter()
            .find(|record| record.plan.trial_id == planned.trial_id)
            .ok_or(ResearchError::Invalid(
                "selection seal is missing a planned trial",
            ))?;
        let relative = record.result_file.as_deref().ok_or(ResearchError::Invalid(
            "selection seal includes a failed trial",
        ))?;
        let bytes = fs::read(output.join(relative))?;
        let result: TuningResult = serde_json::from_slice(&bytes)?;
        validate_search_result(protocol, planned, &result)?;
        digest.update(planned.trial_id.as_bytes());
        digest.update(result.search.selected_candidate_id.as_bytes());
        digest.update(Sha256::digest(bytes));
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn correctness_evidence(
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
) -> Result<Vec<String>, ResearchError> {
    let evidence = state
        .search
        .iter()
        .filter(|record| record.plan.workload == workload && record.status == "complete")
        .map(|record| record.plan.trial_id.clone())
        .collect::<Vec<_>>();
    if evidence.len() != 4 || !state.selections_sealed {
        return Err(ResearchError::Invalid(
            "reference evaluation requires four sealed correctness-admitted search trials",
        ));
    }
    Ok(evidence)
}

fn reference_path(workload: [usize; 3], candidate: &str) -> String {
    format!(
        "references/m{}k{}n{}/{}.json",
        workload[0], workload[1], workload[2], candidate
    )
}

fn validate_reference_identity(
    record: &CandidateReferenceRecord,
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
    candidate: &str,
) -> Result<(), ResearchError> {
    if record.schema != REFERENCE_RECORD_SCHEMA
        || record.session_id != state.session_id
        || record.session_index != state.session_index
        || record.protocol_sha256 != state.protocol_sha256
        || record.source_revision != state.software.source_revision
        || record.source_tree_sha256 != state.software.source_tree_sha256
        || record.workload != workload
        || record.candidate_id != candidate
        || Some(&record.selection_seal_sha256) != state.selection_seal_sha256.as_ref()
    {
        return Err(ResearchError::Invalid(
            "reference record identity, source, protocol, or selection seal mismatch",
        ));
    }
    Ok(())
}

fn validate_complete_reference(
    protocol: &ProspectiveProtocol,
    record: &CandidateReferenceRecord,
) -> Result<(), ResearchError> {
    if record.blocks.len() != protocol.reference_blocks
        || record
            .blocks
            .iter()
            .map(|block| block.block_index)
            .collect::<BTreeSet<_>>()
            .len()
            != protocol.reference_blocks
        || record.blocks.iter().any(|block| {
            block.candidate_id != record.candidate_id
                || block.raw_samples_ns.len() != protocol.reference_runs_per_block as usize
                || validate_samples(&block.raw_samples_ns).is_err()
        })
    {
        return Err(ResearchError::Invalid(
            "candidate reference record has missing, duplicate, or invalid blocks",
        ));
    }
    Ok(())
}

fn validate_complete_session(
    protocol: &ProspectiveProtocol,
    state: &ProspectiveSessionRecord,
    output: &Path,
) -> Result<(), ResearchError> {
    if !state.selections_sealed
        || state.software.research_implementation_version.as_deref()
            != Some(CORRECTED_IMPLEMENTATION_VERSION)
        || state.search.len() != 8
        || state.reference_files.len() != protocol.workloads.len() * 10
        || state.reference_attempts.len()
            != protocol.workloads.len() * tiled_candidate_ids().len() * protocol.reference_blocks
        || state
            .reference_attempts
            .iter()
            .any(|attempt| attempt.status != "completed_and_durably_recorded")
        || state
            .reference_attempts
            .iter()
            .map(|attempt| (attempt.workload, &attempt.candidate_id, attempt.block_index))
            .collect::<BTreeSet<_>>()
            .len()
            != state.reference_attempts.len()
        || state
            .search
            .iter()
            .map(|record| &record.plan.trial_id)
            .collect::<BTreeSet<_>>()
            .len()
            != state.search.len()
    {
        return Err(ResearchError::Invalid("prospective session is incomplete"));
    }
    let mut session_adapter: Option<AdapterSummaryRecord> = None;
    for planned in &state.plan.search_trials {
        let record = state
            .search
            .iter()
            .find(|record| record.plan == *planned && record.status == "complete")
            .ok_or(ResearchError::Invalid(
                "prospective session has a missing or unexpected search trial",
            ))?;
        let relative = record.result_file.as_ref().ok_or(ResearchError::Invalid(
            "complete search trial lacks a result file",
        ))?;
        let result: TuningResult = serde_json::from_slice(&fs::read(output.join(relative))?)?;
        validate_search_result(protocol, planned, &result)?;
        if result.software.source_revision != state.software.source_revision
            || result.software.source_tree_sha256 != state.software.source_tree_sha256
        {
            return Err(ResearchError::Invalid(
                "search result source identity mismatch",
            ));
        }
        if session_adapter
            .as_ref()
            .is_some_and(|saved| !same_adapter(saved, &result.adapter))
        {
            return Err(ResearchError::Invalid(
                "search adapter changed within prospective session",
            ));
        }
        session_adapter = Some(result.adapter);
    }
    if selection_seal(output, state, protocol)?
        != state
            .selection_seal_sha256
            .clone()
            .ok_or(ResearchError::Invalid("selection seal is missing"))?
    {
        return Err(ResearchError::Invalid(
            "selection seal no longer matches search records",
        ));
    }
    let expected = tiled_candidate_ids().into_iter().collect::<BTreeSet<_>>();
    let expected_reference_files = protocol
        .workloads
        .iter()
        .flat_map(|workload| {
            expected
                .iter()
                .map(move |candidate| reference_path(*workload, candidate))
        })
        .collect::<BTreeSet<_>>();
    if state.reference_files.iter().collect::<BTreeSet<_>>()
        != expected_reference_files.iter().collect::<BTreeSet<_>>()
        || super::find_json(&output.join("references"))?.len() != expected_reference_files.len()
        || super::find_json(&output.join("trials"))?.len() != state.plan.search_trials.len()
    {
        return Err(ResearchError::Invalid("unexpected reference record path"));
    }
    for workload in &protocol.workloads {
        let mut found = BTreeSet::new();
        for candidate in &expected {
            let path = output.join(reference_path(*workload, candidate));
            let record: CandidateReferenceRecord = serde_json::from_slice(&fs::read(path)?)?;
            validate_reference_identity(&record, state, *workload, candidate)?;
            validate_complete_reference(protocol, &record)?;
            let adapter = record.adapter.as_ref().ok_or(ResearchError::Invalid(
                "complete reference record lacks adapter",
            ))?;
            if !approved_hardware(adapter)
                || session_adapter
                    .as_ref()
                    .is_some_and(|saved| !same_adapter(saved, adapter))
            {
                return Err(ResearchError::Invalid(
                    "reference adapter differs from the approved session hardware",
                ));
            }
            for block in &record.blocks {
                if !state.reference_attempts.iter().any(|attempt| {
                    attempt.workload == *workload
                        && attempt.candidate_id == *candidate
                        && attempt.block_index == block.block_index
                        && attempt.status == "completed_and_durably_recorded"
                }) {
                    return Err(ResearchError::Invalid(
                        "reference block lacks a completed durable invocation intent",
                    ));
                }
                let planned = state
                    .plan
                    .reference_blocks
                    .iter()
                    .find(|planned| {
                        planned.workload == *workload && planned.block_index == block.block_index
                    })
                    .ok_or(ResearchError::Invalid(
                        "reference block is not in the frozen plan",
                    ))?;
                if planned.candidate_ids.get(block.order_index) != Some(candidate) {
                    return Err(ResearchError::Invalid(
                        "reference candidate block order differs from the frozen plan",
                    ));
                }
            }
            if record.status != "complete"
                || record.correctness_evidence_trial_ids != correctness_evidence(state, *workload)?
                || !found.insert(candidate)
            {
                return Err(ResearchError::Invalid(
                    "prospective reference panel is incomplete or duplicated",
                ));
            }
        }
    }
    Ok(())
}

fn same_adapter(left: &AdapterSummaryRecord, right: &AdapterSummaryRecord) -> bool {
    left.name == right.name
        && left.backend == right.backend
        && left.device_type == right.device_type
        && left.timestamp_query_supported == right.timestamp_query_supported
        && left.timestamp_period_ns == right.timestamp_period_ns
}

fn approved_hardware(adapter: &AdapterSummaryRecord) -> bool {
    adapter.name == "Apple M5"
        && adapter.backend.eq_ignore_ascii_case("metal")
        && adapter.timestamp_query_supported
        && adapter
            .timestamp_period_ns
            .is_some_and(|period| period.is_finite() && period > 0.0)
}

fn validate_samples(samples: &[f64]) -> Result<(), ResearchError> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ResearchError::Invalid(
            "GPU timestamp samples must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_output_path(path: &Path) -> Result<(), ResearchError> {
    if path.as_os_str().is_empty()
        || path == Path::new("/")
        || path
            .components()
            .any(|component| component == Component::ParentDir)
    {
        return Err(ResearchError::Invalid("unsafe prospective output path"));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProspectiveValidation {
    pub schema: String,
    pub valid: bool,
    pub sessions: usize,
    pub search_trials: usize,
    pub reference_records: usize,
    pub errors: Vec<String>,
}

pub fn validate_prospective_results(input: &Path) -> Result<ProspectiveValidation, ResearchError> {
    let mut report = ProspectiveValidation {
        schema: "constraintforge-m7-prospective-validation-v1".into(),
        valid: true,
        sessions: 0,
        search_trials: 0,
        reference_records: 0,
        errors: Vec::new(),
    };
    let mut session_ids = BTreeSet::new();
    let mut session_indices = BTreeSet::new();
    let mut protocol_identity = None;
    let mut source_identity = None;
    for path in super::find_named(input, "session.json")? {
        let state: ProspectiveSessionRecord = serde_json::from_slice(&fs::read(&path)?)?;
        let root = path
            .parent()
            .ok_or(ResearchError::Invalid("invalid session path"))?;
        let protocol = ProspectiveProtocol::parse(&root.join("resolved-protocol.toml"))?;
        let result = validate_complete_session(&protocol, &state, root);
        if state.schema != PROSPECTIVE_SESSION_SCHEMA
            || state.phase != "complete"
            || !session_ids.insert(state.session_id.clone())
            || !session_indices.insert(state.session_index)
            || protocol_identity.get_or_insert(state.protocol_sha256.clone())
                != &state.protocol_sha256
            || source_identity.get_or_insert((
                state.software.source_revision.clone(),
                state.software.source_tree_sha256.clone(),
            )) != &(
                state.software.source_revision.clone(),
                state.software.source_tree_sha256.clone(),
            )
        {
            report
                .errors
                .push(format!("invalid or duplicate session: {}", path.display()));
        }
        if let Err(error) = result {
            report.errors.push(format!("{}: {error}", path.display()));
        }
        report.sessions += 1;
        report.search_trials += state.search.len();
        report.reference_records += state.reference_files.len();
    }
    if report.sessions == 0 {
        report.errors.push("no prospective sessions found".into());
    }
    report.valid = report.errors.is_empty();
    Ok(report)
}

pub fn write_prospective_manifest(input: &Path, output: &Path) -> Result<usize, ResearchError> {
    if output.exists() {
        return Err(ResearchError::OutputExists(output.into()));
    }
    let validation = validate_prospective_results(input)?;
    if !validation.valid
        || validation.sessions != 5
        || validation.search_trials != 40
        || validation.reference_records != 100
    {
        return Err(ResearchError::Invalid(
            "prospective manifest requires the complete validated five-session pilot",
        ));
    }
    let mut paths = Vec::new();
    for session in super::find_named(input, "session.json")? {
        let root = session
            .parent()
            .ok_or(ResearchError::Invalid("invalid prospective session path"))?;
        let before = paths.len();
        collect_files(root, &mut paths)?;
        if paths.len() - before != 30 {
            return Err(ResearchError::Invalid(
                "prospective session must contain 30 evidence files including its protocol copy",
            ));
        }
    }
    paths.sort();
    let mut manifest = String::new();
    for path in &paths {
        let relative = path.strip_prefix(input).map_err(|_| {
            ResearchError::Invalid("prospective manifest path escapes the input root")
        })?;
        manifest.push_str(&format!(
            "{:x}  {}\n",
            Sha256::digest(fs::read(path)?),
            relative.to_string_lossy().replace('\\', "/")
        ));
    }
    write_new(output, manifest.trim_end().as_bytes())?;
    Ok(paths.len())
}

fn collect_files(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), ResearchError> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, paths)?;
        } else if path.is_file() {
            paths.push(path);
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProspectiveAnalysis {
    pub schema: String,
    pub classification: String,
    pub sessions: usize,
    pub rows: usize,
    pub source_files: Vec<String>,
}

#[derive(Clone, Debug)]
struct ReferenceSummary {
    candidate: String,
    median30: f64,
    prefixes: [f64; 3],
    block_medians: [f64; 3],
    rank: f64,
}

pub fn analyze_prospective(
    input: &Path,
    output: &Path,
) -> Result<ProspectiveAnalysis, ResearchError> {
    analyze_prospective_mode(input, output, 5, "PROSPECTIVE PILOT / DESCRIPTIVE")
}

pub fn analyze_prospective_qualification(
    input: &Path,
    output: &Path,
) -> Result<ProspectiveAnalysis, ResearchError> {
    analyze_prospective_mode(input, output, 1, "QUALIFICATION / DESCRIPTIVE ONLY")
}

fn analyze_prospective_mode(
    input: &Path,
    output: &Path,
    required_sessions: usize,
    classification: &str,
) -> Result<ProspectiveAnalysis, ResearchError> {
    if output.exists() {
        return Err(ResearchError::OutputExists(output.into()));
    }
    let validation = validate_prospective_results(input)?;
    if !validation.valid || validation.sessions != required_sessions {
        return Err(ResearchError::Invalid(
            "prospective analysis has the wrong complete-session count for this mode",
        ));
    }
    fs::create_dir(output)?;
    let mut states = Vec::new();
    for path in super::find_named(input, "session.json")? {
        let state: ProspectiveSessionRecord = serde_json::from_slice(&fs::read(&path)?)?;
        states.push((path.parent().unwrap().to_path_buf(), state));
    }
    states.sort_by_key(|(_, state)| state.session_index);
    let mut regret = String::from(
        "session,session_index,m,k,n,budget,policy,selected_candidate,reference_median_ns,empirical_min_ns,relative_regret,reference_rank,best_agreement,shared_reference_panel,statistical_unit,interpretation_limit\n",
    );
    let mut stability = String::from(
        "session,session_index,m,k,n,candidate,median10_ns,median20_ns,median30_ns,block0_median_ns,block1_median_ns,block2_median_ns,rank30,empirical_best_10,empirical_best_20,empirical_best_30,statistical_unit,interpretation_limit\n",
    );
    let mut comparison = String::from(
        "session,session_index,m,k,n,budget,resource_aware_regret,random_regret,paired_difference,statistical_unit,interpretation_limit\n",
    );
    let mut rank_stability = String::from(
        "session,session_index,m,k,n,left_block,right_block,shared_candidates,spearman_rho,kendall_tau_b,empirical_best_agreement,statistical_unit,interpretation_limit\n",
    );
    let mut costs = String::from(
        "session,session_index,m,k,n,budget,policy,search_score_ns,reference_median_ns,search_reference_ratio,search_wall_time_ns,tuning_wall_time_ns,decision_overhead_ns,statistical_unit,interpretation_limit\n",
    );
    let mut rows = 0;
    let mut source_files = Vec::new();
    for (root, state) in &states {
        let protocol = ProspectiveProtocol::parse(&root.join("resolved-protocol.toml"))?;
        for workload in &protocol.workloads {
            let summaries = load_reference_summaries(root, state, *workload, &mut source_files)?;
            let minima = [0, 1, 2].map(|index| {
                summaries
                    .iter()
                    .map(|summary| summary.prefixes[index])
                    .min_by(f64::total_cmp)
                    .unwrap()
            });
            for summary in &summaries {
                stability.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},candidate within session/workload,empirical reference has measurement error and shares one physical session\n",
                    state.session_id,
                    state.session_index,
                    workload[0], workload[1], workload[2],
                    summary.candidate,
                    summary.prefixes[0], summary.prefixes[1], summary.prefixes[2],
                    summary.block_medians[0], summary.block_medians[1], summary.block_medians[2],
                    summary.rank,
                    summary.prefixes[0] == minima[0],
                    summary.prefixes[1] == minima[1],
                    summary.prefixes[2] == minima[2],
                ));
            }
            for left in 0..3 {
                for right in left + 1..3 {
                    let left_values = summaries
                        .iter()
                        .map(|summary| summary.block_medians[left])
                        .collect::<Vec<_>>();
                    let right_values = summaries
                        .iter()
                        .map(|summary| summary.block_medians[right])
                        .collect::<Vec<_>>();
                    let (spearman, kendall) = rank_agreement(&left_values, &right_values)?;
                    let left_best = left_values
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
                        .unwrap()
                        .0;
                    let right_best = right_values
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
                        .unwrap()
                        .0;
                    rank_stability.push_str(&format!(
                        "{},{},{},{},{},{},{},{},{},{},{},candidate ranks within session/workload,three block comparisons are descriptive and dependent\n",
                        state.session_id, state.session_index,
                        workload[0], workload[1], workload[2],
                        left, right, summaries.len(), spearman, kendall, left_best == right_best,
                    ));
                }
            }
            let empirical_min = summaries
                .iter()
                .map(|summary| summary.median30)
                .min_by(f64::total_cmp)
                .unwrap();
            let mut paired = BTreeMap::<usize, BTreeMap<String, f64>>::new();
            for record in state
                .search
                .iter()
                .filter(|record| record.plan.workload == *workload)
            {
                let relative = record.result_file.as_ref().unwrap();
                source_files.push(root.join(relative).display().to_string());
                let result: TuningResult = serde_json::from_slice(&fs::read(root.join(relative))?)?;
                let selected = summaries
                    .iter()
                    .find(|summary| summary.candidate == result.search.selected_candidate_id)
                    .ok_or(ResearchError::Invalid(
                        "selected candidate is missing from reference panel",
                    ))?;
                let relative_regret = relative_regret(selected.median30, empirical_min)?;
                let best_agreement = selected.median30 == empirical_min;
                regret.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},true,session,reference denominator is shared across matched policies and budgets\n",
                    state.session_id, state.session_index,
                    workload[0], workload[1], workload[2],
                    record.plan.budget, record.plan.policy_id,
                    selected.candidate, selected.median30, empirical_min, relative_regret,
                    selected.rank, best_agreement,
                ));
                let search_score =
                    result
                        .search
                        .selected_score_ns
                        .ok_or(ResearchError::Invalid(
                            "measured policy lacks selected search score",
                        ))?;
                costs.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},trial nested in session,same-candidate disagreement is not oracle regret\n",
                    state.session_id, state.session_index,
                    workload[0], workload[1], workload[2],
                    record.plan.budget, record.plan.policy_id,
                    search_score, selected.median30, search_score / selected.median30,
                    result.search_wall_time_ns, result.tuning_wall_time_ns,
                    result.search_decision_overhead_ns,
                ));
                paired
                    .entry(record.plan.budget)
                    .or_default()
                    .insert(record.plan.policy_id.clone(), relative_regret);
                rows += 1;
            }
            for (budget, values) in paired {
                let aware = values
                    .get("resource-aware")
                    .ok_or(ResearchError::Invalid("missing resource-aware regret"))?;
                let random = values
                    .get("random")
                    .ok_or(ResearchError::Invalid("missing random regret"))?;
                comparison.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},paired session/workload/budget,descriptive prospective evidence; no confidence interval p-value or superiority claim\n",
                    state.session_id, state.session_index,
                    workload[0], workload[1], workload[2], budget,
                    aware, random, aware - random,
                ));
            }
        }
    }
    write_new(&output.join("regret.csv"), regret.trim_end().as_bytes())?;
    write_new(
        &output.join("reference-stability.csv"),
        stability.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("paired-policy.csv"),
        comparison.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("reference-rank-agreement.csv"),
        rank_stability.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("search-reference-cost.csv"),
        costs.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("validation.json"),
        &serde_json::to_vec_pretty(&validation)?,
    )?;
    source_files.sort();
    source_files.dedup();
    let analysis = ProspectiveAnalysis {
        schema: "constraintforge-m7-prospective-analysis-v1".into(),
        classification: classification.into(),
        sessions: states.len(),
        rows,
        source_files,
    };
    write_new(
        &output.join("analysis.json"),
        &serde_json::to_vec_pretty(&analysis)?,
    )?;
    write_analysis_manifest(output)?;
    Ok(analysis)
}

fn load_reference_summaries(
    root: &Path,
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
    source_files: &mut Vec<String>,
) -> Result<Vec<ReferenceSummary>, ResearchError> {
    let mut raw = Vec::new();
    for candidate in tiled_candidate_ids() {
        let path = root.join(reference_path(workload, &candidate));
        source_files.push(path.display().to_string());
        let record: CandidateReferenceRecord = serde_json::from_slice(&fs::read(path)?)?;
        validate_reference_identity(&record, state, workload, &candidate)?;
        let mut samples = Vec::new();
        let mut block_medians = [0.0; 3];
        for block in &record.blocks {
            block_medians[block.block_index] = median(&block.raw_samples_ns)?;
            samples.extend_from_slice(&block.raw_samples_ns);
        }
        raw.push((
            candidate,
            median(&samples[..10])?,
            median(&samples[..20])?,
            median(&samples[..30])?,
            block_medians,
        ));
    }
    let ranks = average_ranks(&raw.iter().map(|entry| entry.3).collect::<Vec<_>>());
    Ok(raw
        .into_iter()
        .zip(ranks)
        .map(
            |((candidate, p10, p20, p30, blocks), rank)| ReferenceSummary {
                candidate,
                median30: p30,
                prefixes: [p10, p20, p30],
                block_medians: blocks,
                rank,
            },
        )
        .collect())
}

fn median(values: &[f64]) -> Result<f64, ResearchError> {
    validate_samples(values)?;
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn relative_regret(selected: f64, empirical_min: f64) -> Result<f64, ResearchError> {
    if !selected.is_finite()
        || !empirical_min.is_finite()
        || selected <= 0.0
        || empirical_min <= 0.0
    {
        return Err(ResearchError::Invalid(
            "regret requires finite positive reference latency",
        ));
    }
    Ok((selected - empirical_min) / empirical_min)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then(left.cmp(right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]].total_cmp(&values[order[start]]).is_eq() {
            end += 1;
        }
        let rank = ((start + 1 + end) as f64) / 2.0;
        for index in &order[start..end] {
            ranks[*index] = rank;
        }
        start = end;
    }
    ranks
}

fn rank_agreement(left: &[f64], right: &[f64]) -> Result<(f64, f64), ResearchError> {
    if left.len() != right.len() || left.len() < 2 {
        return Err(ResearchError::Invalid(
            "rank agreement requires equal candidate coverage",
        ));
    }
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    let left_mean = left_ranks.iter().sum::<f64>() / left_ranks.len() as f64;
    let right_mean = right_ranks.iter().sum::<f64>() / right_ranks.len() as f64;
    let covariance = left_ranks
        .iter()
        .zip(&right_ranks)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let scale = (left_ranks
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        * right_ranks
            .iter()
            .map(|value| (value - right_mean).powi(2))
            .sum::<f64>())
    .sqrt();
    if scale == 0.0 {
        return Err(ResearchError::Invalid(
            "rank agreement is undefined for constant ranks",
        ));
    }
    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut left_ties = 0.0_f64;
    let mut right_ties = 0.0_f64;
    for i in 0..left.len() {
        for j in i + 1..left.len() {
            let a = left[i].total_cmp(&left[j]);
            let b = right[i].total_cmp(&right[j]);
            match (a.is_eq(), b.is_eq()) {
                (true, true) => {}
                (true, false) => left_ties += 1.0,
                (false, true) => right_ties += 1.0,
                (false, false) if a == b => concordant += 1.0,
                (false, false) => discordant += 1.0,
            }
        }
    }
    let denominator: f64 =
        ((concordant + discordant + left_ties) * (concordant + discordant + right_ties)).sqrt();
    if denominator == 0.0 {
        return Err(ResearchError::Invalid("Kendall tau-b is undefined"));
    }
    Ok((covariance / scale, (concordant - discordant) / denominator))
}

fn write_analysis_manifest(output: &Path) -> Result<(), ResearchError> {
    let mut paths = fs::read_dir(output)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut manifest = String::new();
    for path in paths {
        if path.file_name().is_some_and(|name| name == "SHA256SUMS") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy();
        manifest.push_str(&format!("{:x}  {name}\n", Sha256::digest(fs::read(&path)?)));
    }
    write_new(&output.join("SHA256SUMS"), manifest.trim_end().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use constraintforge_bench::Statistics;
    use constraintforge_tuner::{
        AdapterSummaryRecord, CorrectnessAdmission, SearchDecision, SearchOutcome,
    };

    fn protocol() -> ProspectiveProtocol {
        toml::from_str(include_str!("../../../research/protocols/m7-phase2a.toml")).unwrap()
    }

    #[test]
    fn frozen_counts_and_equal_effort_are_exact() {
        let protocol = protocol();
        protocol.validate().unwrap();
        assert_eq!(protocol.expected_search_trials, 40);
        assert_eq!(protocol.expected_reference_records, 100);
        assert_eq!(protocol.expected_reference_blocks, 300);
        assert_eq!(protocol.expected_total_timed_samples, 4_360);
        assert_eq!(protocol.expected_total_candidate_invocations, 6_100);
    }

    #[test]
    fn plans_are_deterministic_balanced_and_complete() {
        let protocol = protocol();
        let left = prospective_plan(&protocol, 0).unwrap();
        let right = prospective_plan(&protocol, 0).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.search_trials.len(), 8);
        assert_eq!(left.reference_blocks.len(), 6);
        for block in &left.reference_blocks {
            assert_eq!(block.candidate_ids.len(), 10);
            assert_eq!(
                block.candidate_ids.iter().collect::<BTreeSet<_>>().len(),
                10
            );
        }
        assert_ne!(
            prospective_plan(&protocol, 0).unwrap(),
            prospective_plan(&protocol, 1).unwrap()
        );
    }

    #[test]
    fn reference_prefixes_ties_and_regret_direction_are_fixed() {
        assert_eq!(median(&[1.0, 3.0, 2.0, 4.0]).unwrap(), 2.5);
        assert_eq!(relative_regret(10.0, 8.0).unwrap(), 0.25);
        assert_eq!(relative_regret(8.0, 8.0).unwrap(), 0.0);
        assert_eq!(average_ranks(&[1.0, 1.0, 3.0]), [1.5, 1.5, 3.0]);
    }

    #[test]
    fn invalid_samples_and_missing_reference_are_not_imputed() {
        assert!(median(&[0.0, 1.0]).is_err());
        assert!(median(&[f64::NAN]).is_err());
        assert!(relative_regret(1.0, 0.0).is_err());
    }

    #[test]
    fn unsafe_and_existing_outputs_are_rejected() {
        assert!(validate_output_path(Path::new("../escape")).is_err());
        assert!(validate_output_path(Path::new("/")).is_err());
    }

    #[test]
    fn sealing_requires_all_search_conditions() {
        let protocol = protocol();
        let plan = prospective_plan(&protocol, 0).unwrap();
        let state = ProspectiveSessionRecord {
            schema: PROSPECTIVE_SESSION_SCHEMA.into(),
            session_id: "test".into(),
            session_index: 0,
            protocol_sha256: "x".into(),
            software: current_software_metadata().unwrap(),
            plan,
            phase: "search".into(),
            search: Vec::new(),
            selections_sealed: false,
            selection_seal_sha256: None,
            reference_files: Vec::new(),
            reference_attempts: Vec::new(),
            environment: Vec::new(),
            source_preflight_captured_utc: 0,
            generated_output_created_after_preflight: true,
            started_utc: 0,
            ended_utc: None,
            interruptions: Vec::new(),
        };
        assert!(selection_seal(Path::new("missing"), &state, &protocol).is_err());
        assert!(correctness_evidence(&state, [7, 19, 11]).is_err());
    }

    #[test]
    fn completed_reference_requires_all_unique_blocks() {
        let protocol = protocol();
        let record = CandidateReferenceRecord {
            schema: REFERENCE_RECORD_SCHEMA.into(),
            session_id: "s".into(),
            session_index: 0,
            protocol_sha256: "p".into(),
            source_revision: "r".into(),
            source_tree_sha256: Some("t".into()),
            workload: [7, 19, 11],
            candidate_id: "c".into(),
            correctness_evidence_trial_ids: vec!["a".into(); 4],
            selection_seal_sha256: "seal".into(),
            adapter: None,
            blocks: vec![ReferenceBlockResult {
                block_index: 0,
                order_index: 0,
                candidate_id: "c".into(),
                started_utc: 0,
                ended_utc: 1,
                raw_samples_ns: vec![1.0; 10],
            }],
            preparation_time_ns: 0,
            measurement_wall_time_ns: 0,
            status: "incomplete".into(),
            error: None,
        };
        assert!(validate_complete_reference(&protocol, &record).is_err());
    }

    fn test_adapter() -> AdapterSummaryRecord {
        AdapterSummaryRecord {
            name: "Apple M5".into(),
            backend: "Metal".into(),
            device_type: "test".into(),
            max_buffer_size: 1,
            max_storage_buffer_binding_size: 1,
            max_compute_workgroups_per_dimension: 1,
            timestamp_query_supported: true,
            timestamp_period_ns: Some(1.0),
        }
    }

    fn synthetic_result(
        protocol: &ProspectiveProtocol,
        plan: &PlannedSearchTrial,
        software: &SoftwareMetadata,
        selected: &str,
    ) -> TuningResult {
        let policy = if plan.policy_id == "random" {
            Policy::Random
        } else {
            Policy::ResourceAware
        };
        let history = (0..protocol.max_batches)
            .map(|batch_index| SearchDecision {
                batch_index,
                candidate_id: selected.into(),
                reason: "synthetic fixture".into(),
                raw_samples_ns: vec![10.0, 11.0, 12.0],
                median_ns: Some(11.0),
                correctness_passed: true,
                error: None,
            })
            .collect();
        TuningResult {
            schema_version: TUNING_SCHEMA_VERSION.into(),
            trial_id: plan.trial_id.clone(),
            utc_unix_seconds: 1,
            shape_mkn: plan.workload,
            input_seed: plan.seed,
            search_seed: plan.seed,
            policy_version: "synthetic".into(),
            search_design: Some(SearchDesign::MatchedBatchesV2),
            measurement_mode: MeasurementMode::GpuTimestamp,
            budget_max_candidates: plan.budget,
            budget_max_batches: protocol.max_batches,
            budget_max_duration_ms: protocol.max_duration_ms,
            adapter: test_adapter(),
            software: software.clone(),
            candidate_pool: Vec::new(),
            static_rejections: Vec::new(),
            correctness_admission: tiled_candidate_ids()
                .into_iter()
                .map(|candidate_id| CorrectnessAdmission {
                    candidate_id,
                    passed: true,
                    maximum_absolute_error: 0.0,
                    maximum_relative_error: 0.0,
                    mismatch_count: 0,
                })
                .collect(),
            search: SearchOutcome {
                policy,
                selected_candidate_id: selected.into(),
                selected_score_ns: Some(11.0),
                candidates_considered: plan.budget,
                candidates_measured: plan.budget,
                measurement_batches: protocol.max_batches,
                total_measured_dispatches: protocol.max_batches * protocol.tuning_runs as usize,
                search_decision_overhead_ns: 1,
                ordered_history: history,
                termination_reason: "budget exhausted".into(),
            },
            independent_evaluation_samples_ns: vec![12.0; 10],
            independent_evaluation_statistics_ns: Statistics::calculate(&[12.0; 10]).unwrap(),
            preparation_time_ns: 1,
            correctness_validation_time_ns: 1,
            search_wall_time_ns: 2,
            tuning_wall_time_ns: 3,
            search_decision_overhead_ns: 1,
            compiled_candidates: 10,
            complete: true,
            errors: Vec::new(),
        }
    }

    fn fixture_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "constraintforge-{name}-{}-{}",
            std::process::id(),
            now().unwrap()
        ))
    }

    fn write_synthetic_session(
        root: &Path,
        protocol: &ProspectiveProtocol,
        session_index: usize,
    ) -> ProspectiveSessionRecord {
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("resolved-protocol.toml"),
            toml::to_string_pretty(protocol).unwrap(),
        )
        .unwrap();
        let plan = prospective_plan(protocol, session_index).unwrap();
        let mut software = current_software_metadata().unwrap();
        software.research_implementation_version = Some(CORRECTED_IMPLEMENTATION_VERSION.into());
        let candidates = tiled_candidate_ids();
        let mut search = Vec::new();
        for trial in &plan.search_trials {
            let selected = if trial.policy_id == "resource-aware" {
                &candidates[0]
            } else {
                &candidates[1]
            };
            let relative = format!("trials/{}.json", trial.trial_id);
            let result = synthetic_result(protocol, trial, &software, selected);
            write_new(
                &root.join(&relative),
                &serde_json::to_vec_pretty(&result).unwrap(),
            )
            .unwrap();
            search.push(ProspectiveSearchRecord {
                plan: trial.clone(),
                status: "complete".into(),
                result_file: Some(relative),
                error: None,
            });
        }
        let mut state = ProspectiveSessionRecord {
            schema: PROSPECTIVE_SESSION_SCHEMA.into(),
            session_id: format!("m7-phase2a-session-{:03}", session_index + 1),
            session_index,
            protocol_sha256: protocol.sha256().unwrap(),
            software,
            plan,
            phase: "reference".into(),
            search,
            selections_sealed: false,
            selection_seal_sha256: None,
            reference_files: Vec::new(),
            reference_attempts: Vec::new(),
            environment: Vec::new(),
            source_preflight_captured_utc: 1,
            generated_output_created_after_preflight: true,
            started_utc: 1,
            ended_utc: None,
            interruptions: Vec::new(),
        };
        state.selection_seal_sha256 = Some(selection_seal(root, &state, protocol).unwrap());
        state.selections_sealed = true;
        for workload in &protocol.workloads {
            for (candidate_index, candidate) in candidates.iter().enumerate() {
                let relative = reference_path(*workload, candidate);
                let base = 10.0 + candidate_index as f64;
                let record = CandidateReferenceRecord {
                    schema: REFERENCE_RECORD_SCHEMA.into(),
                    session_id: state.session_id.clone(),
                    session_index,
                    protocol_sha256: state.protocol_sha256.clone(),
                    source_revision: state.software.source_revision.clone(),
                    source_tree_sha256: state.software.source_tree_sha256.clone(),
                    workload: *workload,
                    candidate_id: candidate.clone(),
                    correctness_evidence_trial_ids: correctness_evidence(&state, *workload)
                        .unwrap(),
                    selection_seal_sha256: state.selection_seal_sha256.clone().unwrap(),
                    adapter: Some(test_adapter()),
                    blocks: (0..3)
                        .map(|block_index| ReferenceBlockResult {
                            block_index,
                            order_index: state
                                .plan
                                .reference_blocks
                                .iter()
                                .find(|planned| {
                                    planned.workload == *workload
                                        && planned.block_index == block_index
                                })
                                .unwrap()
                                .candidate_ids
                                .iter()
                                .position(|id| id == candidate)
                                .unwrap(),
                            candidate_id: candidate.clone(),
                            started_utc: block_index as u64,
                            ended_utc: block_index as u64 + 1,
                            raw_samples_ns: vec![base + block_index as f64; 10],
                        })
                        .collect(),
                    preparation_time_ns: 1,
                    measurement_wall_time_ns: 1,
                    status: "complete".into(),
                    error: None,
                };
                write_new(
                    &root.join(&relative),
                    &serde_json::to_vec_pretty(&record).unwrap(),
                )
                .unwrap();
                state.reference_files.push(relative);
                for block_index in 0..protocol.reference_blocks {
                    state.reference_attempts.push(ReferenceAttemptRecord {
                        workload: *workload,
                        candidate_id: candidate.clone(),
                        block_index,
                        status: "completed_and_durably_recorded".into(),
                        started_utc: 1,
                        ended_utc: Some(2),
                        error: None,
                    });
                }
            }
        }
        state.reference_files.sort();
        state.phase = "complete".into();
        state.ended_utc = Some(2);
        atomic_json(&root.join("session.json"), &state).unwrap();
        state
    }

    #[test]
    fn synthetic_validation_rejects_duplicates_and_missing_reference() {
        let root = fixture_root("prospective-validation");
        let protocol = protocol();
        let session = root.join("session-001");
        let state = write_synthetic_session(&session, &protocol, 0);
        assert!(validate_prospective_results(&root).unwrap().valid);

        let missing = session.join(&state.reference_files[0]);
        fs::remove_file(&missing).unwrap();
        assert!(!validate_prospective_results(&root).unwrap().valid);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn completed_reference_orphans_reconcile_without_remeasurement() {
        let root = fixture_root("prospective-reference-reconcile");
        let protocol = protocol();
        let mut state = write_synthetic_session(&root, &protocol, 0);
        state.reference_files.clear();
        reconcile_reference_records(&protocol, &root, &mut state).unwrap();
        assert_eq!(state.reference_files.len(), 20);
        assert_eq!(
            state
                .interruptions
                .iter()
                .filter(|reason| reason.contains("reconciled immutable"))
                .count(),
            20
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reference_protocol_and_seal_mismatches_are_rejected() {
        let root = fixture_root("prospective-reference-identity");
        let protocol = protocol();
        let state = write_synthetic_session(&root, &protocol, 0);
        let path = root.join(&state.reference_files[0]);
        let mut record: CandidateReferenceRecord =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record.protocol_sha256 = "wrong".into();
        atomic_json(&path, &record).unwrap();
        assert!(!validate_prospective_results(&root).unwrap().valid);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn synthetic_analysis_is_deterministic_and_session_paired() {
        let root = fixture_root("prospective-analysis");
        let protocol = protocol();
        for session_index in 0..protocol.sessions {
            write_synthetic_session(
                &root.join(format!("session-{:03}", session_index + 1)),
                &protocol,
                session_index,
            );
        }
        let study_manifest = root.join("full-study-artifacts.sha256");
        assert_eq!(
            write_prospective_manifest(&root, &study_manifest).unwrap(),
            150
        );
        assert!(write_prospective_manifest(&root, &study_manifest).is_err());
        let first = root.join("analysis-a");
        let second = root.join("analysis-b");
        let a = analyze_prospective(&root, &first).unwrap();
        let b = analyze_prospective(&root, &second).unwrap();
        assert_eq!(a.rows, 40);
        assert_eq!(a.sessions, 5);
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        for name in [
            "analysis.json",
            "paired-policy.csv",
            "reference-rank-agreement.csv",
            "reference-stability.csv",
            "regret.csv",
            "search-reference-cost.csv",
            "validation.json",
            "SHA256SUMS",
        ] {
            assert_eq!(
                fs::read(first.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        assert!(analyze_prospective(&root, &first).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn production_sequence_initializes_reference_storage_and_qualifies_one_session() {
        let root = fixture_root("prospective-production-sequence");
        let session = root.join("session-001-requal-01");
        let protocol = protocol();
        let mut software = current_software_metadata().unwrap();
        software.git_worktree_clean = Some(true);
        software.research_implementation_version = Some(CORRECTED_IMPLEMENTATION_VERSION.into());
        let search_software = software.clone();
        let reference_software = software.clone();
        let search_calls = std::cell::Cell::new(0);
        let reference_calls = std::cell::Cell::new(0);
        let state = run_prospective_session_with(
            ProspectiveRunRequest {
                protocol: &protocol,
                session_id: "m7-phase2a-session-001-requal-01",
                session_index: 0,
                output: &session,
                resume: false,
            },
            || Ok(software.clone()),
            |config| {
                search_calls.set(search_calls.get() + 1);
                let plan = prospective_plan(&protocol, 0)
                    .unwrap()
                    .search_trials
                    .into_iter()
                    .find(|plan| {
                        plan.workload == config.shape
                            && plan.budget == config.budget.max_candidates
                            && ((plan.policy_id == "random" && config.policy == Policy::Random)
                                || (plan.policy_id == "resource-aware"
                                    && config.policy == Policy::ResourceAware))
                    })
                    .unwrap();
                Ok(synthetic_result(
                    &protocol,
                    &plan,
                    &search_software,
                    &tiled_candidate_ids()[0],
                ))
            },
            |config| {
                reference_calls.set(reference_calls.get() + 1);
                assert!(
                    session
                        .join(reference_path(config.shape, &config.candidate_id))
                        .parent()
                        .unwrap()
                        .is_dir()
                );
                Ok((
                    test_adapter(),
                    reference_software.clone(),
                    ReferenceBlockResult {
                        block_index: config.block_index,
                        order_index: config.order_index,
                        candidate_id: config.candidate_id.clone(),
                        started_utc: 1,
                        ended_utc: 2,
                        raw_samples_ns: vec![10.0 + config.order_index as f64; 10],
                    },
                    1,
                    1,
                ))
            },
        )
        .unwrap();
        assert_eq!(state.phase, "complete");
        assert_eq!(search_calls.get(), 8);
        assert_eq!(reference_calls.get(), 60);
        assert_eq!(state.reference_files.len(), 20);
        assert_eq!(state.reference_attempts.len(), 60);
        assert!(state.software.git_worktree_clean.unwrap());
        assert!(validate_prospective_results(&session).unwrap().valid);

        let first = root.join("qualification-a");
        let second = root.join("qualification-b");
        let a = analyze_prospective_qualification(&session, &first).unwrap();
        let b = analyze_prospective_qualification(&session, &second).unwrap();
        assert_eq!(a.classification, "QUALIFICATION / DESCRIPTIVE ONLY");
        assert_eq!(a.sessions, 1);
        assert_eq!(a.rows, 8);
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        for name in [
            "analysis.json",
            "paired-policy.csv",
            "reference-rank-agreement.csv",
            "reference-stability.csv",
            "regret.csv",
            "search-reference-cost.csv",
            "validation.json",
            "SHA256SUMS",
        ] {
            assert_eq!(
                fs::read(first.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        assert!(analyze_prospective(&session, &root.join("full-analysis")).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ambiguous_reference_attempt_is_never_remeasured() {
        let root = fixture_root("prospective-ambiguous-attempt");
        let protocol = protocol();
        let mut state = write_synthetic_session(&root, &protocol, 0);
        let attempt = &mut state.reference_attempts[0];
        attempt.status = "started_or_possibly_executed".into();
        attempt.ended_utc = None;
        let path = root.join(reference_path(attempt.workload, &attempt.candidate_id));
        let mut record: CandidateReferenceRecord =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record
            .blocks
            .retain(|block| block.block_index != attempt.block_index);
        atomic_json(&path, &record).unwrap();
        assert!(reconcile_reference_attempts(&root, &mut state).is_err());
        assert_eq!(
            state.reference_attempts[0].status,
            "started_or_possibly_executed"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reference_storage_failure_occurs_before_any_reference_invocation() {
        let root = fixture_root("prospective-storage-failure");
        let session = root.join("session");
        let protocol = protocol();
        let plan = prospective_plan(&protocol, 0).unwrap();
        let mut software = current_software_metadata().unwrap();
        software.git_worktree_clean = Some(false);
        software.research_implementation_version = Some(CORRECTED_IMPLEMENTATION_VERSION.into());
        let search_software = software.clone();
        let search_calls = std::cell::Cell::new(0);
        let reference_calls = std::cell::Cell::new(0);
        let result = run_prospective_session_with(
            ProspectiveRunRequest {
                protocol: &protocol,
                session_id: "storage-failure",
                session_index: 0,
                output: &session,
                resume: false,
            },
            || Ok(software.clone()),
            |config| {
                search_calls.set(search_calls.get() + 1);
                let trial = plan
                    .search_trials
                    .iter()
                    .find(|trial| {
                        trial.workload == config.shape
                            && trial.budget == config.budget.max_candidates
                            && ((trial.policy_id == "random" && config.policy == Policy::Random)
                                || (trial.policy_id == "resource-aware"
                                    && config.policy == Policy::ResourceAware))
                    })
                    .unwrap();
                if search_calls.get() == 8 {
                    fs::write(session.join("references"), b"not a directory").unwrap();
                }
                Ok(synthetic_result(
                    &protocol,
                    trial,
                    &search_software,
                    &tiled_candidate_ids()[0],
                ))
            },
            |_| {
                reference_calls.set(reference_calls.get() + 1);
                Err("reference operation must not run".into())
            },
        );
        assert!(result.is_err());
        assert_eq!(search_calls.get(), 8);
        assert_eq!(reference_calls.get(), 0);
        let state: ProspectiveSessionRecord =
            serde_json::from_slice(&fs::read(session.join("session.json")).unwrap()).unwrap();
        assert_eq!(state.software.git_worktree_clean, Some(false));
        assert!(state.selections_sealed);
        assert!(state.reference_attempts.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn durable_reference_commit_reconciles_before_checkpoint() {
        let root = fixture_root("prospective-committed-attempt");
        let protocol = protocol();
        let mut state = write_synthetic_session(&root, &protocol, 0);
        state.reference_attempts[0].status = "started_or_possibly_executed".into();
        state.reference_attempts[0].ended_utc = None;
        reconcile_reference_attempts(&root, &mut state).unwrap();
        assert_eq!(
            state.reference_attempts[0].status,
            "completed_and_durably_recorded"
        );
        fs::remove_dir_all(root).unwrap();
    }
}

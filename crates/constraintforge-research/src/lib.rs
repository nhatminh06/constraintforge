use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use constraintforge_bench::{SoftwareMetadata, current_software_metadata};
use constraintforge_tuner::{
    DEVELOPMENT_SHAPES, GpuTuningConfig, HELD_OUT_SHAPES, MeasurementMode, Policy, ScoreAblation,
    SearchBudget, SearchDesign, TUNING_SCHEMA_VERSION, TuningResult, run_gpu,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

mod robustness;
pub use robustness::{RobustnessAnalysis, robustness};
mod reference_robustness;
pub use reference_robustness::{ReferenceReliabilityAnalysis, reference_robustness};
mod reference_panels;
pub use reference_panels::{
    ReferencePanelAnalysis, ReferencePanelPlan, ReferencePanelProtocol, ReferencePanelSession,
    ReferencePanelValidation, analyze_reference_panels, reference_panel_plan,
    run_reference_panel_session, validate_reference_panel_results,
};
mod prospective;
pub use prospective::{
    ProspectiveAnalysis, ProspectivePlan, ProspectiveProtocol, ProspectiveSessionRecord,
    ProspectiveValidation, analyze_prospective, analyze_prospective_qualification,
    prospective_plan, run_prospective_session, validate_prospective_results,
    write_prospective_manifest,
};

pub const PROTOCOL_SCHEMA: &str = "constraintforge-research-protocol-v1";
pub const SESSION_SCHEMA: &str = "constraintforge-research-session-v1";
pub const ANALYSIS_SCHEMA: &str = "constraintforge-research-analysis-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Protocol {
    pub schema: String,
    pub id: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_design: Option<String>,
    pub phase: String,
    pub frozen: bool,
    pub research_question: String,
    pub hardware_target: String,
    pub candidate_space: String,
    pub measurement: String,
    pub ordering: String,
    pub result_schema: String,
    pub analysis_method: String,
    pub stop_rules: String,
    pub development_workloads: Vec<[usize; 3]>,
    pub held_out_workloads: Vec<[usize; 3]>,
    pub active_split: String,
    pub workload_limit: usize,
    pub policies: Vec<String>,
    pub budgets: Vec<usize>,
    pub seeds: Vec<u64>,
    pub sessions: usize,
    pub max_batches: usize,
    pub max_duration_ms: u64,
    pub tuning_warmups: u32,
    pub tuning_runs: u32,
    pub evaluation_warmups: u32,
    pub evaluation_runs: u32,
    pub analysis_seed: u64,
    pub bootstrap_resamples: usize,
}

impl Protocol {
    pub fn parse(path: &Path) -> Result<Self, ResearchError> {
        let text = fs::read_to_string(path)?;
        let protocol: Self = toml::from_str(&text)?;
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ResearchError> {
        if self.schema != PROTOCOL_SCHEMA || self.result_schema != TUNING_SCHEMA_VERSION {
            return Err(ResearchError::Invalid(
                "incompatible protocol or result schema",
            ));
        }
        if self.development_workloads != DEVELOPMENT_SHAPES
            || self.held_out_workloads != HELD_OUT_SHAPES
            || self
                .development_workloads
                .iter()
                .any(|x| self.held_out_workloads.contains(x))
        {
            return Err(ResearchError::Invalid(
                "workload split differs from the frozen M6 split",
            ));
        }
        if self.sessions == 0 || self.seeds.len() < self.sessions || self.budgets.is_empty() {
            return Err(ResearchError::Invalid(
                "sessions, seeds, and budgets must be nonzero",
            ));
        }
        if self.max_batches == 0
            || self.max_duration_ms == 0
            || self.tuning_runs == 0
            || self.evaluation_runs == 0
        {
            return Err(ResearchError::Invalid(
                "measurement limits and repetitions must be nonzero",
            ));
        }
        let known = [
            "fixed",
            "heuristic",
            "random",
            "resource-ranked-full",
            "resource-ranked-no-padding",
            "resource-ranked-no-shared",
            "resource-ranked-no-reuse",
            "resource-aware",
            "exhaustive",
        ];
        if self.policies.is_empty() || self.policies.iter().any(|p| !known.contains(&p.as_str())) {
            return Err(ResearchError::Invalid("unknown research policy ID"));
        }
        if self.phase == "frozen" && !self.frozen {
            return Err(ResearchError::Invalid("frozen phase must be immutable"));
        }
        match (
            self.implementation_version.as_deref(),
            self.search_design.as_deref(),
        ) {
            (None, None) => {}
            (Some("m6-v2"), Some("matched-batches-v2")) => {}
            _ => {
                return Err(ResearchError::Invalid(
                    "unsupported research implementation or search design",
                ));
            }
        }
        if self.search_design.as_deref() == Some("matched-batches-v2") {
            if self.max_batches != 8
                || self.tuning_runs != 3
                || self
                    .budgets
                    .iter()
                    .any(|budget| *budget == 0 || *budget > 8)
                || !self.policies.iter().any(|policy| policy == "random")
                || !self
                    .policies
                    .iter()
                    .any(|policy| policy == "resource-aware")
            {
                return Err(ResearchError::Invalid(
                    "M6-v2 requires random/resource-aware, 8 batches, 3 samples, and candidate budgets in 1..=8",
                ));
            }
            if self.phase == "frozen"
                && (self.sessions != 5
                    || self.workload_limit != 5
                    || self.policies.len() != 9
                    || self.budgets != [2, 4, 6, 8]
                    || self.seeds[..self.sessions] != [11, 23, 37, 41, 53])
            {
                return Err(ResearchError::Invalid(
                    "frozen M6-v2 design differs from the approved 900-trial plan",
                ));
            }
        }
        if self.active_split != "development" && self.active_split != "held-out" {
            return Err(ResearchError::Invalid(
                "active_split must be development or held-out",
            ));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, ResearchError> {
        let bytes = serde_json::to_vec(self)?;
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        Ok(format!("fnv1a64:{hash:016x}"))
    }

    pub fn sha256(&self) -> Result<String, ResearchError> {
        Ok(format!(
            "{:x}",
            Sha256::digest(toml::to_string_pretty(self)?)
        ))
    }

    pub fn workloads(&self) -> &[[usize; 3]] {
        let workloads = if self.active_split == "held-out" {
            &self.held_out_workloads
        } else {
            &self.development_workloads
        };
        &workloads[..self.workload_limit.min(workloads.len())]
    }

    fn search_design(&self) -> SearchDesign {
        if self.search_design.as_deref() == Some("matched-batches-v2") {
            SearchDesign::MatchedBatchesV2
        } else {
            SearchDesign::Legacy
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialRecord {
    pub trial_id: String,
    pub workload: [usize; 3],
    pub budget: usize,
    pub policy_id: String,
    pub seed: u64,
    pub order_index: usize,
    pub status: String,
    pub result_file: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    pub schema: String,
    pub session_id: String,
    #[serde(default)]
    pub session_index: usize,
    pub protocol_fingerprint: String,
    #[serde(default)]
    pub protocol_sha256: Option<String>,
    pub started_utc: u64,
    pub ended_utc: Option<u64>,
    pub status: String,
    pub software: SoftwareMetadata,
    pub execution_order: Vec<String>,
    pub trials: Vec<TrialRecord>,
    pub interruptions: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RunControl {
    pub stop_after_persisted_trials: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub schema: String,
    pub valid: bool,
    pub sessions: usize,
    pub complete_trials: usize,
    pub incomplete_trials: usize,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub input_files: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregateRow {
    pub workload: [usize; 3],
    pub budget: usize,
    pub policy_id: String,
    pub independent_units: usize,
    pub median_latency_ns: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairedComparison {
    pub workload: [usize; 3],
    pub budget: usize,
    pub policy_id: String,
    pub pairs: usize,
    pub geometric_latency_ratio_to_random: f64,
    pub ci95: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analysis {
    pub schema: String,
    pub analysis_seed: u64,
    pub bootstrap_resamples: usize,
    pub source_files: Vec<String>,
    pub aggregates: Vec<AggregateRow>,
    pub paired: Vec<PairedComparison>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialHash {
    pub trial_id: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResumeCheckpoint {
    pub schema: String,
    pub session_id: String,
    pub protocol_sha256: String,
    pub source_revision: String,
    pub source_tree_sha256: Option<String>,
    pub controlled_stop_reason: String,
    pub trials: Vec<TrialHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResumeQualificationReport {
    pub schema: String,
    pub checkpoint: ResumeCheckpoint,
    pub post_resume_trials: Vec<TrialHash>,
    pub hashes_unchanged: bool,
    pub expected_trials: usize,
    pub final_status: String,
    pub validation: ValidationReport,
    pub analysis_rows: usize,
}

pub fn policy_order(
    protocol: &Protocol,
    session_index: usize,
    condition_index: usize,
) -> Vec<String> {
    let mut policies = protocol.policies.clone();
    if !policies.is_empty() {
        let shift = (session_index + condition_index) % policies.len();
        policies.rotate_left(shift);
        if session_index % 2 == 1 {
            policies.reverse();
        }
    }
    policies
}

pub fn inspect(protocol: &Protocol) -> Result<Vec<String>, ResearchError> {
    let mut trials = Vec::new();
    for session in 0..protocol.sessions {
        let seed = protocol.seeds[session];
        for (wi, workload) in protocol.workloads().iter().enumerate() {
            for (bi, budget) in protocol.budgets.iter().enumerate() {
                for policy in policy_order(protocol, session, wi * protocol.budgets.len() + bi) {
                    trials.push(trial_id(session, *workload, *budget, &policy, seed));
                }
            }
        }
    }
    Ok(trials)
}

pub fn run_session(
    protocol: &Protocol,
    session_id: &str,
    session_index: usize,
    output: &Path,
    resume: bool,
) -> Result<SessionRecord, ResearchError> {
    run_session_controlled(
        protocol,
        session_id,
        session_index,
        output,
        resume,
        RunControl::default(),
    )
}

pub fn qualify_resume(
    protocol: &Protocol,
    session_id: &str,
    output: &Path,
    evidence: &Path,
    analysis_output: &Path,
) -> Result<ResumeQualificationReport, ResearchError> {
    if output.exists() || evidence.exists() || analysis_output.exists() {
        return Err(ResearchError::Invalid(
            "qualification output, evidence, and analysis paths must not exist",
        ));
    }
    fs::create_dir(evidence)?;
    sync_parent(evidence)?;
    match run_session_controlled(
        protocol,
        session_id,
        0,
        output,
        false,
        RunControl {
            stop_after_persisted_trials: Some(2),
        },
    ) {
        Err(ResearchError::ControlledStop(2)) => {}
        Ok(_) => {
            return Err(ResearchError::Invalid(
                "qualification did not produce the expected controlled stop",
            ));
        }
        Err(error) => return Err(error),
    }
    let state: SessionRecord = serde_json::from_slice(&fs::read(output.join("session.json"))?)?;
    let reason = "controlled stop after 2 persisted trials";
    if state.status != "interrupted"
        || state.trials.len() != 2
        || state.trials.iter().any(|trial| trial.status != "complete")
        || state.interruptions.last().map(String::as_str) != Some(reason)
    {
        return Err(ResearchError::Invalid(
            "controlled-stop state does not contain exactly two durable complete trials",
        ));
    }
    let pre_resume = trial_hashes(output, &state.trials)?;
    if pre_resume.len() != 2 {
        return Err(ResearchError::Invalid(
            "controlled-stop checkpoint does not contain two unique trial files",
        ));
    }
    let checkpoint = ResumeCheckpoint {
        schema: "constraintforge-resume-checkpoint-v1".into(),
        session_id: session_id.into(),
        protocol_sha256: protocol.sha256()?,
        source_revision: state.software.source_revision.clone(),
        source_tree_sha256: state.software.source_tree_sha256.clone(),
        controlled_stop_reason: reason.into(),
        trials: pre_resume,
    };
    durable_json(&evidence.join("checkpoint.json"), &checkpoint)?;

    let final_state = run_session(protocol, session_id, 0, output, true)?;
    if final_state.trials.len() < 2 {
        return Err(ResearchError::Invalid(
            "resumed session lost checkpointed trials",
        ));
    }
    let post_resume = trial_hashes(output, &final_state.trials[..2])?;
    let hashes_unchanged = checkpoint
        .trials
        .iter()
        .map(|trial| (&trial.trial_id, &trial.sha256))
        .eq(post_resume
            .iter()
            .map(|trial| (&trial.trial_id, &trial.sha256)));
    if !hashes_unchanged {
        return Err(ResearchError::Invalid(
            "completed trial changed across resume",
        ));
    }
    let expected_trials = planned_session(protocol, 0).len();
    if final_state.status != "complete"
        || final_state.trials.len() != expected_trials
        || final_state
            .trials
            .iter()
            .map(|trial| &trial.trial_id)
            .collect::<BTreeSet<_>>()
            .len()
            != expected_trials
    {
        return Err(ResearchError::Invalid(
            "resumed session is incomplete or contains duplicate trials",
        ));
    }
    let validation = validate_results(output)?;
    if !validation.valid || validation.complete_trials != expected_trials {
        return Err(ResearchError::Invalid(
            "resumed session failed final validation",
        ));
    }
    let analysis = analyze(
        output,
        analysis_output,
        protocol.analysis_seed,
        protocol.bootstrap_resamples,
    )?;
    let report = ResumeQualificationReport {
        schema: "constraintforge-resume-qualification-v1".into(),
        checkpoint,
        post_resume_trials: post_resume,
        hashes_unchanged,
        expected_trials,
        final_status: final_state.status,
        validation,
        analysis_rows: analysis.aggregates.len(),
    };
    durable_json(&evidence.join("report.json"), &report)?;
    Ok(report)
}

pub fn run_session_controlled(
    protocol: &Protocol,
    session_id: &str,
    session_index: usize,
    output: &Path,
    resume: bool,
    control: RunControl,
) -> Result<SessionRecord, ResearchError> {
    protocol.validate()?;
    if session_index >= protocol.sessions {
        return Err(ResearchError::Invalid("session index exceeds protocol"));
    }
    fs::create_dir_all(output)?;
    let manifest_path = output.join("resolved-protocol.toml");
    let fingerprint = protocol.fingerprint()?;
    let state_path = output.join("session.json");
    if manifest_path.exists() || state_path.exists() {
        if !resume {
            return Err(ResearchError::OutputExists(output.to_path_buf()));
        }
        let saved = Protocol::parse(&manifest_path)?;
        let state: SessionRecord = serde_json::from_slice(&fs::read(&state_path)?)?;
        if saved.fingerprint()? != fingerprint
            || state.protocol_fingerprint != fingerprint
            || state.session_id != session_id
            || state.session_index != session_index
            || state.software.source_fingerprint_fnv1a64
                != software_for(protocol)?.source_fingerprint_fnv1a64
        {
            return Err(ResearchError::Invalid(
                "resume protocol or source provenance mismatch",
            ));
        }
    } else {
        write_new(&manifest_path, toml::to_string_pretty(protocol)?.as_bytes())?;
    }
    let seed = protocol.seeds[session_index];
    let mut state = if state_path.exists() {
        serde_json::from_slice(&fs::read(&state_path)?)?
    } else {
        SessionRecord {
            schema: SESSION_SCHEMA.into(),
            session_id: session_id.into(),
            session_index,
            protocol_fingerprint: fingerprint,
            protocol_sha256: Some(protocol.sha256()?),
            started_utc: now()?,
            ended_utc: None,
            status: "running".into(),
            software: software_for(protocol)?,
            execution_order: vec![],
            trials: vec![],
            interruptions: vec![],
        }
    };
    atomic_json(&state_path, &state)?;
    reconcile_orphan_results(protocol, session_index, output, &mut state)?;
    atomic_json(&state_path, &state)?;
    let completed: BTreeSet<_> = state
        .trials
        .iter()
        .filter(|t| t.status == "complete")
        .map(|t| t.trial_id.clone())
        .collect();
    for (wi, workload) in protocol.workloads().iter().enumerate() {
        for (bi, budget) in protocol.budgets.iter().enumerate() {
            let order = policy_order(protocol, session_index, wi * protocol.budgets.len() + bi);
            for (order_index, policy_id) in order.iter().enumerate() {
                let id = trial_id(session_index, *workload, *budget, policy_id, seed);
                if completed.contains(&id) {
                    continue;
                }
                if state.trials.iter().any(|trial| trial.trial_id == id) {
                    return Err(ResearchError::Invalid(
                        "duplicate incomplete trial requires a new session",
                    ));
                }
                state.execution_order.push(id.clone());
                atomic_json(&state_path, &state)?;
                let policy = policy(policy_id)?;
                let (max_candidates, max_batches) = search_limits(protocol, policy, *budget);
                let score_ablation = match policy_id.as_str() {
                    "resource-ranked-no-padding" => ScoreAblation::NoPadding,
                    "resource-ranked-no-shared" => ScoreAblation::NoShared,
                    "resource-ranked-no-reuse" => ScoreAblation::NoReuse,
                    _ => ScoreAblation::Full,
                };
                let run = run_gpu(&GpuTuningConfig {
                    shape: *workload,
                    workload_seed: seed,
                    search_seed: seed,
                    policy,
                    score_ablation,
                    search_design: protocol.search_design(),
                    budget: SearchBudget {
                        max_candidates,
                        max_batches,
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
                        result.trial_id = id.clone();
                        result.software = software_for(protocol)?;
                        if protocol.search_design() == SearchDesign::MatchedBatchesV2 {
                            if policy == Policy::Random {
                                result.policy_version = "m6-v2-random-round-robin-v1".into();
                            } else if policy == Policy::ResourceAware {
                                result.policy_version = "m6-v2-resource-aware-v1".into();
                            } else if policy == Policy::ResourceRankedOnly {
                                result.policy_version =
                                    "m6-v2-resource-ranked-round-robin-v1".into();
                            }
                        }
                        let name = format!("trials/{id}.json");
                        write_new(&output.join(&name), &serde_json::to_vec_pretty(&result)?)?;
                        ("complete".into(), Some(name), None)
                    }
                    Err(error) => ("failed".into(), None, Some(error.to_string())),
                };
                state.trials.push(TrialRecord {
                    trial_id: id,
                    workload: *workload,
                    budget: *budget,
                    policy_id: policy_id.clone(),
                    seed,
                    order_index,
                    status,
                    result_file,
                    error,
                });
                atomic_json(&state_path, &state)?;
                if control.stop_after_persisted_trials == Some(state.trials.len()) {
                    state.status = "interrupted".into();
                    state.ended_utc = Some(now()?);
                    state.interruptions.push(format!(
                        "controlled stop after {} persisted trials",
                        state.trials.len()
                    ));
                    atomic_json(&state_path, &state)?;
                    return Err(ResearchError::ControlledStop(state.trials.len()));
                }
            }
        }
    }
    state.status = if state.trials.iter().all(|t| t.status == "complete") {
        "complete"
    } else {
        "complete-with-failures"
    }
    .into();
    state.ended_utc = Some(now()?);
    atomic_json(&state_path, &state)?;
    Ok(state)
}

fn reconcile_orphan_results(
    protocol: &Protocol,
    session_index: usize,
    output: &Path,
    state: &mut SessionRecord,
) -> Result<(), ResearchError> {
    let seed = protocol.seeds[session_index];
    for (wi, workload) in protocol.workloads().iter().enumerate() {
        for (bi, budget) in protocol.budgets.iter().enumerate() {
            for (order_index, policy_id) in
                policy_order(protocol, session_index, wi * protocol.budgets.len() + bi)
                    .iter()
                    .enumerate()
            {
                let id = trial_id(session_index, *workload, *budget, policy_id, seed);
                if state.trials.iter().any(|trial| trial.trial_id == id) {
                    continue;
                }
                let relative = format!("trials/{id}.json");
                let path = output.join(&relative);
                if path.exists() {
                    let result = read_valid_result(&path)?;
                    if result.trial_id != id
                        || result.shape_mkn != *workload
                        || result.search_seed != seed
                        || result.software.source_fingerprint_fnv1a64
                            != state.software.source_fingerprint_fnv1a64
                    {
                        return Err(ResearchError::Invalid(
                            "orphan trial identity or provenance mismatch",
                        ));
                    }
                    state.execution_order.push(id.clone());
                    state.trials.push(TrialRecord {
                        trial_id: id,
                        workload: *workload,
                        budget: *budget,
                        policy_id: policy_id.clone(),
                        seed,
                        order_index,
                        status: "complete".into(),
                        result_file: Some(relative),
                        error: None,
                    });
                    state
                        .interruptions
                        .push("reconciled immutable trial persisted before session update".into());
                }
            }
        }
    }
    Ok(())
}

pub fn validate_results(input: &Path) -> Result<ValidationReport, ResearchError> {
    let mut report = ValidationReport {
        schema: "constraintforge-research-validation-v1".into(),
        valid: true,
        sessions: 0,
        complete_trials: 0,
        incomplete_trials: 0,
        errors: vec![],
        warnings: vec![],
        input_files: vec![],
    };
    let mut session_ids = BTreeSet::new();
    for state_path in find_named(input, "session.json")? {
        report.sessions += 1;
        report.input_files.push(state_path.display().to_string());
        let state: SessionRecord = match serde_json::from_slice(&fs::read(&state_path)?) {
            Ok(x) => x,
            Err(e) => {
                report.errors.push(format!("{}: {e}", state_path.display()));
                continue;
            }
        };
        if !session_ids.insert(state.session_id.clone()) {
            report
                .errors
                .push(format!("duplicate session identity {}", state.session_id));
        }
        let session_dir = state_path.parent().unwrap_or(input);
        let manifest_path = session_dir.join("resolved-protocol.toml");
        let protocol = match Protocol::parse(&manifest_path) {
            Ok(protocol) => protocol,
            Err(error) => {
                report
                    .errors
                    .push(format!("{}: {error}", manifest_path.display()));
                continue;
            }
        };
        if state.schema != SESSION_SCHEMA
            || state.protocol_fingerprint != protocol.fingerprint()?
            || state.session_index >= protocol.sessions
        {
            report.errors.push(format!(
                "session {} has incompatible schema, protocol, or index",
                state.session_id
            ));
        }
        if let Some(protocol_sha256) = &state.protocol_sha256 {
            if protocol_sha256 != &protocol.sha256()? {
                report.errors.push(format!(
                    "session {} protocol SHA-256 mismatch",
                    state.session_id
                ));
            }
        } else {
            report.warnings.push(format!(
                "session {} predates protocol SHA-256 metadata",
                state.session_id
            ));
        }
        let expected = planned_session(&protocol, state.session_index);
        let mut ids = BTreeSet::new();
        for trial in &state.trials {
            if !ids.insert(trial.trial_id.clone()) {
                report
                    .errors
                    .push(format!("duplicate trial {}", trial.trial_id));
            }
            let Some(relative) = &trial.result_file else {
                report.incomplete_trials += 1;
                report.warnings.push(format!(
                    "trial {} status {} has no result",
                    trial.trial_id, trial.status
                ));
                continue;
            };
            if Path::new(relative).is_absolute()
                || Path::new(relative)
                    .components()
                    .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                report
                    .errors
                    .push(format!("trial {} has unsafe result path", trial.trial_id));
                continue;
            }
            let path = session_dir.join(relative);
            report.input_files.push(path.display().to_string());
            match read_valid_result(&path) {
                Ok(result)
                    if result.trial_id == trial.trial_id
                        && result.shape_mkn == trial.workload
                        && result.search_seed == trial.seed
                        && result.software.source_fingerprint_fnv1a64
                            == state.software.source_fingerprint_fnv1a64 =>
                {
                    if let Err(error) = validate_trial_result(&protocol, trial, &result) {
                        report.errors.push(format!("{}: {error}", path.display()));
                    } else {
                        report.complete_trials += 1
                    }
                }
                Ok(_) => report.errors.push(format!(
                    "{}: trial identity, shape, seed, or source mismatch",
                    path.display()
                )),
                Err(e) => report.errors.push(format!("{}: {e}", path.display())),
            }
        }
        for missing in expected.difference(&ids) {
            report
                .errors
                .push(format!("missing required trial {missing}"));
        }
        for extra in ids.difference(&expected) {
            report.errors.push(format!("unexpected trial {extra}"));
        }
        let recorded_files = state
            .trials
            .iter()
            .filter_map(|trial| trial.result_file.as_deref())
            .map(|relative| session_dir.join(relative))
            .collect::<BTreeSet<_>>();
        for file in find_json(&session_dir.join("trials"))? {
            if !recorded_files.contains(&file) {
                report
                    .errors
                    .push(format!("unreconciled trial file {}", file.display()));
            }
        }
        if state.status != "complete" {
            report.warnings.push(format!(
                "session {} status is {}",
                state.session_id, state.status
            ));
        }
    }
    if report.sessions == 0 {
        report.errors.push("no session records found".into());
    }
    report.valid = report.errors.is_empty();
    Ok(report)
}

pub fn analyze(
    input: &Path,
    output: &Path,
    seed: u64,
    resamples: usize,
) -> Result<Analysis, ResearchError> {
    let validation = validate_results(input)?;
    if !validation.valid {
        return Err(ResearchError::Invalid("input validation failed"));
    }
    if output.exists() {
        return Err(ResearchError::OutputExists(output.to_path_buf()));
    }
    let mut units: BTreeMap<([usize; 3], usize, String), Vec<f64>> = BTreeMap::new();
    let mut paired_units: BTreeMap<(String, [usize; 3], usize, String), f64> = BTreeMap::new();
    let mut sources = Vec::new();
    let mut strata = BTreeSet::new();
    for state_path in find_named(input, "session.json")? {
        let state: SessionRecord = serde_json::from_slice(&fs::read(&state_path)?)?;
        for trial in state.trials.iter().filter(|t| t.status == "complete") {
            let path = state_path
                .parent()
                .unwrap_or(input)
                .join(trial.result_file.as_ref().unwrap());
            let result = read_valid_result(&path)?;
            strata.insert((
                state.protocol_fingerprint.clone(),
                result
                    .software
                    .source_tree_sha256
                    .clone()
                    .unwrap_or_else(|| result.software.source_fingerprint_fnv1a64.clone()),
                result
                    .software
                    .research_implementation_version
                    .clone()
                    .unwrap_or_else(|| "unversioned".into()),
                result.adapter.name.clone(),
                result.adapter.backend.clone(),
            ));
            let latency = median(&result.independent_evaluation_samples_ns)
                .ok_or(ResearchError::Invalid("missing independent evaluation"))?;
            units
                .entry((trial.workload, trial.budget, trial.policy_id.clone()))
                .or_default()
                .push(latency);
            paired_units.insert(
                (
                    state.session_id.clone(),
                    trial.workload,
                    trial.budget,
                    trial.policy_id.clone(),
                ),
                latency,
            );
            sources.push(path.display().to_string());
        }
    }
    if strata.len() > 1 {
        return Err(ResearchError::Invalid(
            "analysis input mixes protocol, source, implementation, or hardware strata",
        ));
    }
    let aggregates = units
        .iter()
        .map(|((workload, budget, policy_id), values)| AggregateRow {
            workload: *workload,
            budget: *budget,
            policy_id: policy_id.clone(),
            independent_units: values.len(),
            median_latency_ns: median(values).unwrap(),
        })
        .collect::<Vec<_>>();
    let mut warnings = validation.warnings.clone();
    let mut paired = Vec::new();
    for ((workload, budget, policy_id), _) in units.iter().filter(|((_, _, p), _)| p != "random") {
        let mut logs = Vec::new();
        for ((session, shape, b, p), value) in &paired_units {
            if shape == workload && b == budget && p == policy_id {
                if let Some(random) =
                    paired_units.get(&(session.clone(), *shape, *b, "random".into()))
                {
                    logs.push((value / random).ln());
                }
            }
        }
        if logs.is_empty() {
            continue;
        }
        if logs.len() < 3 {
            warnings.push(format!(
                "insufficient replication for {policy_id} {:?} budget {budget}",
                workload
            ));
        }
        let ci95 = (logs.len() >= 3 && resamples > 0).then(|| bootstrap_ci(&logs, seed, resamples));
        paired.push(PairedComparison {
            workload: *workload,
            budget: *budget,
            policy_id: policy_id.clone(),
            pairs: logs.len(),
            geometric_latency_ratio_to_random: mean(&logs).exp(),
            ci95: ci95.map(|[a, b]| [a.exp(), b.exp()]),
        });
    }
    sources.sort();
    let analysis = Analysis {
        schema: ANALYSIS_SCHEMA.into(),
        analysis_seed: seed,
        bootstrap_resamples: resamples,
        source_files: sources,
        aggregates,
        paired,
        warnings,
    };
    fs::create_dir_all(output)?;
    write_new(
        &output.join("aggregate.json"),
        &serde_json::to_vec_pretty(&analysis)?,
    )?;
    write_new(
        &output.join("latency-vs-budget.csv"),
        csv(&analysis).as_bytes(),
    )?;
    write_new(&output.join("summary.md"), summary(&analysis).as_bytes())?;
    write_new(
        &output.join("latency-vs-budget.svg"),
        svg(&analysis).as_bytes(),
    )?;
    write_new(
        &output.join("validation.json"),
        &serde_json::to_vec_pretty(&validation)?,
    )?;
    Ok(analysis)
}

fn policy(id: &str) -> Result<Policy, ResearchError> {
    Ok(match id {
        "fixed" => Policy::Fixed,
        "heuristic" => Policy::Heuristic,
        "random" => Policy::Random,
        x if x.starts_with("resource-ranked-") => Policy::ResourceRankedOnly,
        "resource-aware" => Policy::ResourceAware,
        "exhaustive" => Policy::Exhaustive,
        _ => return Err(ResearchError::Invalid("unknown policy")),
    })
}

fn search_limits(protocol: &Protocol, policy: Policy, budget: usize) -> (usize, usize) {
    let max_candidates = if policy == Policy::Exhaustive {
        10
    } else if matches!(policy, Policy::Fixed | Policy::Heuristic) {
        0
    } else {
        budget
    };
    let max_batches = if matches!(policy, Policy::Fixed | Policy::Heuristic) {
        0
    } else if policy == Policy::Exhaustive {
        10
    } else if policy == Policy::ResourceRankedOnly
        && protocol.search_design() == SearchDesign::Legacy
    {
        max_candidates.min(protocol.max_batches)
    } else {
        protocol.max_batches
    };
    (max_candidates, max_batches)
}

fn planned_session(protocol: &Protocol, session_index: usize) -> BTreeSet<String> {
    let seed = protocol.seeds[session_index];
    protocol
        .workloads()
        .iter()
        .flat_map(|workload| {
            protocol.budgets.iter().flat_map(move |budget| {
                protocol
                    .policies
                    .iter()
                    .map(move |policy| trial_id(session_index, *workload, *budget, policy, seed))
            })
        })
        .collect()
}

fn trial_id(
    session: usize,
    [m, k, n]: [usize; 3],
    budget: usize,
    policy: &str,
    seed: u64,
) -> String {
    format!(
        "s{:02}-m{m}k{k}n{n}-b{budget}-{policy}-seed{seed}",
        session + 1
    )
}
fn now() -> Result<u64, ResearchError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ResearchError::Invalid("system clock before Unix epoch"))?
        .as_secs())
}
fn software_for(protocol: &Protocol) -> Result<SoftwareMetadata, ResearchError> {
    let mut software = current_software_metadata()?;
    if let Some(version) = &protocol.implementation_version {
        software.research_implementation_version = Some(version.clone());
    }
    Ok(software)
}
fn validate_trial_result(
    protocol: &Protocol,
    trial: &TrialRecord,
    result: &TuningResult,
) -> Result<(), String> {
    if result.search.policy != policy(&trial.policy_id).map_err(|error| error.to_string())?
        || result.independent_evaluation_samples_ns.len() != protocol.evaluation_runs as usize
        || !result
            .correctness_admission
            .iter()
            .all(|admission| admission.passed)
        || !result
            .correctness_admission
            .iter()
            .any(|admission| admission.candidate_id == result.search.selected_candidate_id)
    {
        return Err("policy, evaluation, correctness, or selected candidate mismatch".into());
    }
    if protocol.search_design() != SearchDesign::MatchedBatchesV2 {
        return Ok(());
    }
    if result.search_design != Some(SearchDesign::MatchedBatchesV2)
        || result.software.research_implementation_version.as_deref() != Some("m6-v2")
    {
        return Err("missing M6-v2 implementation or matched-batch identity".into());
    }
    if matches!(
        result.search.policy,
        Policy::Random | Policy::ResourceRankedOnly | Policy::ResourceAware
    ) {
        let distinct = result
            .search
            .ordered_history
            .iter()
            .map(|decision| decision.candidate_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        if result.budget_max_candidates != trial.budget
            || result.budget_max_batches != protocol.max_batches
            || result.search.measurement_batches != protocol.max_batches
            || result.search.total_measured_dispatches
                != protocol.max_batches * protocol.tuning_runs as usize
            || distinct > trial.budget
            || result.search.ordered_history.iter().any(|decision| {
                decision.error.is_some()
                    || decision.raw_samples_ns.len() != protocol.tuning_runs as usize
            })
        {
            return Err("M6-v2 unequal or invalid budgeted-search accounting".into());
        }
    }
    Ok(())
}
fn read_valid_result(path: &Path) -> Result<TuningResult, ResearchError> {
    let r: TuningResult = serde_json::from_slice(&fs::read(path)?)?;
    if r.schema_version != TUNING_SCHEMA_VERSION
        || !r.complete
        || r.adapter.name.is_empty()
        || r.software.source_fingerprint_fnv1a64.is_empty()
    {
        return Err(ResearchError::Invalid(
            "missing schema, completion, hardware, or provenance",
        ));
    }
    if r.independent_evaluation_samples_ns.is_empty()
        || r.independent_evaluation_samples_ns
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(ResearchError::Invalid(
            "missing or invalid independent GPU timing",
        ));
    }
    if r.search.ordered_history.iter().any(|b| {
        b.error.is_none()
            && (b.raw_samples_ns.is_empty()
                || b.raw_samples_ns.iter().any(|x| !x.is_finite() || *x <= 0.0))
    }) {
        return Err(ResearchError::Invalid(
            "missing or invalid raw tuning samples",
        ));
    }
    if r.search.measurement_batches > r.budget_max_batches
        || r.search.candidates_measured > r.budget_max_candidates
    {
        return Err(ResearchError::Invalid("record exceeds declared budget"));
    }
    Ok(r)
}
fn find_named(root: &Path, name: &str) -> Result<Vec<PathBuf>, ResearchError> {
    let mut out = Vec::new();
    if root.is_dir() {
        for e in fs::read_dir(root)? {
            let p = e?.path();
            if p.is_dir() {
                out.extend(find_named(&p, name)?)
            } else if p.file_name().is_some_and(|x| x == name) {
                out.push(p)
            }
        }
    }
    out.sort();
    Ok(out)
}
fn find_json(root: &Path) -> Result<Vec<PathBuf>, ResearchError> {
    let mut out = Vec::new();
    if root.is_dir() {
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.is_dir() {
                out.extend(find_json(&path)?);
            } else if path.extension().is_some_and(|value| value == "json") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), ResearchError> {
    if path.exists() {
        return Err(ResearchError::OutputExists(path.into()));
    }
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?
    }
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(bytes)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}
fn trial_hashes(output: &Path, trials: &[TrialRecord]) -> Result<Vec<TrialHash>, ResearchError> {
    let mut ids = BTreeSet::new();
    let mut hashes = Vec::with_capacity(trials.len());
    for trial in trials {
        if !ids.insert(&trial.trial_id) {
            return Err(ResearchError::Invalid(
                "checkpoint contains duplicate trial identity",
            ));
        }
        let relative = trial.result_file.as_deref().ok_or(ResearchError::Invalid(
            "checkpoint trial has no result file",
        ))?;
        let path = output.join(relative);
        hashes.push(TrialHash {
            trial_id: trial.trial_id.clone(),
            path: relative.into(),
            sha256: format!("{:x}", Sha256::digest(fs::read(path)?)),
        });
    }
    Ok(hashes)
}
fn durable_json(path: &Path, value: &impl Serialize) -> Result<(), ResearchError> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    sync_parent(path)?;
    Ok(())
}
fn sync_parent(path: &Path) -> Result<(), ResearchError> {
    if let Some(parent) = path.parent() {
        OpenOptions::new().read(true).open(parent)?.sync_all()?;
    }
    Ok(())
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), ResearchError> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    Some(if x.len() % 2 == 0 {
        (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.0
    } else {
        x[x.len() / 2]
    })
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn bootstrap_ci(values: &[f64], mut state: u64, n: usize) -> [f64; 2] {
    let mut means = Vec::with_capacity(n);
    for _ in 0..n {
        let mut total = 0.0;
        for _ in values {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            total += values[state as usize % values.len()]
        }
        means.push(total / values.len() as f64)
    }
    means.sort_by(f64::total_cmp);
    [means[n * 25 / 1000], means[(n * 975 / 1000).min(n - 1)]]
}
fn csv(a: &Analysis) -> String {
    let mut s = "m,k,n,budget,policy,independent_units,median_latency_ns\n".to_owned();
    for r in &a.aggregates {
        s.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.workload[0],
            r.workload[1],
            r.workload[2],
            r.budget,
            r.policy_id,
            r.independent_units,
            r.median_latency_ns
        ))
    }
    s
}
fn summary(a: &Analysis) -> String {
    format!(
        "# ConstraintForge M6 analysis\n\nValidated independent experimental units only. GPU dispatch samples were aggregated within each trial.\n\nRows: {}. Paired comparisons: {}.\n\n## Warnings\n\n{}\n",
        a.aggregates.len(),
        a.paired.len(),
        if a.warnings.is_empty() {
            "None.".into()
        } else {
            a.warnings
                .iter()
                .map(|x| format!("- {x}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    )
}
fn svg(a: &Analysis) -> String {
    let points=a.aggregates.iter().enumerate().map(|(i,r)|format!("<circle cx=\"{}\" cy=\"{}\" r=\"3\"><title>{} budget {}: {} ns; sources in aggregate.json</title></circle>",40+i*15,180.0-(r.median_latency_ns.log10()*15.0),r.policy_id,r.budget,r.median_latency_ns)).collect::<String>();
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"220\"><title>Independent latency versus budget</title><text x=\"10\" y=\"20\">M6 latency vs budget (trace: aggregate.json source_files)</text><g fill=\"#245\">{points}</g></svg>"
    )
}

#[derive(Debug, Error)]
pub enum ResearchError {
    #[error("invalid research artifact: {0}")]
    Invalid(&'static str),
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
    #[error("controlled stop after {0} persisted trials")]
    ControlledStop(usize),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Bench(#[from] constraintforge_bench::BenchError),
    #[error(transparent)]
    Tuner(#[from] constraintforge_tuner::TunerError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    TomlDecode(#[from] toml::de::Error),
    #[error(transparent)]
    TomlEncode(#[from] toml::ser::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    fn protocol() -> Protocol {
        Protocol {
            schema: PROTOCOL_SCHEMA.into(),
            id: "test".into(),
            version: "1".into(),
            implementation_version: None,
            search_design: None,
            phase: "calibration".into(),
            frozen: false,
            research_question: "q".into(),
            hardware_target: "test".into(),
            candidate_space: "m5-ten".into(),
            measurement: "gpu-timestamp".into(),
            ordering: "counterbalanced".into(),
            result_schema: TUNING_SCHEMA_VERSION.into(),
            analysis_method: "paired-log-bootstrap".into(),
            stop_rules: "limits".into(),
            development_workloads: DEVELOPMENT_SHAPES.into(),
            held_out_workloads: HELD_OUT_SHAPES.into(),
            active_split: "development".into(),
            workload_limit: 6,
            policies: vec![
                "random".into(),
                "resource-ranked-full".into(),
                "resource-aware".into(),
            ],
            budgets: vec![2, 4],
            seeds: vec![11, 23, 37],
            sessions: 3,
            max_batches: 6,
            max_duration_ms: 1000,
            tuning_warmups: 1,
            tuning_runs: 2,
            evaluation_warmups: 1,
            evaluation_runs: 3,
            analysis_seed: 7,
            bootstrap_resamples: 100,
        }
    }
    #[test]
    fn protocol_identity_and_split() {
        let p = protocol();
        p.validate().unwrap();
        assert_eq!(p.fingerprint().unwrap(), p.fingerprint().unwrap());
        assert!(
            DEVELOPMENT_SHAPES
                .iter()
                .all(|x| !HELD_OUT_SHAPES.contains(x))
        );
    }
    #[test]
    fn order_and_seed_assignment_are_reproducible() {
        let p = protocol();
        assert_eq!(policy_order(&p, 1, 2), policy_order(&p, 1, 2));
        let x = inspect(&p).unwrap();
        assert!(x.iter().any(|x| x.contains("seed11")));
        assert_eq!(x.len(), 3 * 6 * 2 * 3);
    }

    #[test]
    fn versioned_protocols_preserve_v1_and_plan_matched_v2() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let v1 = Protocol::parse(&root.join("research/protocols/m6-v1.toml")).unwrap();
        assert_eq!(v1.fingerprint().unwrap(), "fnv1a64:9611e5179c6cc3c5");
        assert_eq!(inspect(&v1).unwrap().len(), 900);

        let v2 = Protocol::parse(&root.join("research/protocols/m6-v2.toml")).unwrap();
        assert_eq!(v2.search_design(), SearchDesign::MatchedBatchesV2);
        assert_eq!(inspect(&v2).unwrap().len(), 900);
        for budget in [2, 4, 6, 8] {
            assert_eq!(search_limits(&v2, Policy::Random, budget), (budget, 8));
            assert_eq!(
                search_limits(&v2, Policy::ResourceAware, budget),
                (budget, 8)
            );
            assert_eq!(
                search_limits(&v2, Policy::ResourceRankedOnly, budget),
                (budget, 8)
            );
        }
        assert_eq!(search_limits(&v2, Policy::Fixed, 2), (0, 0));
        assert_eq!(search_limits(&v2, Policy::Heuristic, 2), (0, 0));
        assert_eq!(search_limits(&v2, Policy::Exhaustive, 2), (10, 10));
    }
    #[test]
    fn bootstrap_is_deterministic() {
        let x = [-0.2, 0.0, 0.2];
        assert_eq!(bootstrap_ci(&x, 42, 1000), bootstrap_ci(&x, 42, 1000));
    }
    #[test]
    fn invalid_schema_and_frozen_state_are_rejected() {
        let mut p = protocol();
        p.schema = "old".into();
        assert!(p.validate().is_err());
        p = protocol();
        p.phase = "frozen".into();
        assert!(p.validate().is_err());
    }

    #[test]
    fn validation_rejects_missing_planned_trials() {
        let mut protocol = protocol();
        protocol.sessions = 1;
        protocol.workload_limit = 1;
        protocol.budgets = vec![2];
        protocol.policies = vec!["random".into(), "resource-aware".into()];
        let directory = std::env::temp_dir().join(format!(
            "constraintforge-research-validation-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("resolved-protocol.toml"),
            toml::to_string_pretty(&protocol).unwrap(),
        )
        .unwrap();
        let mut state = SessionRecord {
            schema: SESSION_SCHEMA.into(),
            session_id: "test".into(),
            session_index: 0,
            protocol_fingerprint: protocol.fingerprint().unwrap(),
            protocol_sha256: Some(protocol.sha256().unwrap()),
            started_utc: 0,
            ended_utc: None,
            status: "interrupted".into(),
            software: current_software_metadata().unwrap(),
            execution_order: vec![],
            trials: vec![],
            interruptions: vec!["test".into()],
        };
        let duplicate = TrialRecord {
            trial_id: "s01-m2k2n2-b2-random-seed11".into(),
            workload: [2, 2, 2],
            budget: 2,
            policy_id: "random".into(),
            seed: 11,
            order_index: 0,
            status: "failed".into(),
            result_file: None,
            error: Some("injected".into()),
        };
        state.trials = vec![duplicate.clone(), duplicate];
        fs::write(
            directory.join("session.json"),
            serde_json::to_vec_pretty(&state).unwrap(),
        )
        .unwrap();
        let report = validate_results(&directory).unwrap();
        assert!(!report.valid);
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("missing required trial"))
        );
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.contains("duplicate trial"))
        );
        state.software.source_fingerprint_fnv1a64 = "different".into();
        fs::write(
            directory.join("session.json"),
            serde_json::to_vec_pretty(&state).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            run_session(&protocol, "test", 0, &directory, true),
            Err(ResearchError::Invalid(
                "resume protocol or source provenance mismatch"
            ))
        ));
        fs::write(directory.join("session.json"), b"{").unwrap();
        let corrupt = validate_results(&directory).unwrap();
        assert!(!corrupt.valid);
        assert!(corrupt.errors.iter().any(|error| error.contains("EOF")));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resume_reconciles_an_immutable_orphan_trial() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let protocol = Protocol::parse(&root.join("research/protocols/m6-pilot.toml")).unwrap();
        let id = "s01-m2k2n2-b2-random-seed11";
        let source = root.join(format!("results/m6-pilot/trials/{id}.json"));
        let result = read_valid_result(&source).unwrap();
        let directory = std::env::temp_dir().join(format!(
            "constraintforge-research-orphan-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(directory.join("trials")).unwrap();
        fs::copy(&source, directory.join(format!("trials/{id}.json"))).unwrap();
        let mut state = SessionRecord {
            schema: SESSION_SCHEMA.into(),
            session_id: "orphan-test".into(),
            session_index: 0,
            protocol_fingerprint: protocol.fingerprint().unwrap(),
            protocol_sha256: None,
            started_utc: 0,
            ended_utc: None,
            status: "running".into(),
            software: result.software,
            execution_order: vec![],
            trials: vec![],
            interruptions: vec![],
        };
        reconcile_orphan_results(&protocol, 0, &directory, &mut state).unwrap();
        assert_eq!(state.trials.len(), 1);
        assert_eq!(state.trials[0].trial_id, id);
        assert!(state.interruptions[0].contains("reconciled immutable trial"));
        assert_eq!(
            fs::read(source).unwrap(),
            fs::read(directory.join(format!("trials/{id}.json"))).unwrap()
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resume_checkpoint_is_durable_unique_and_detects_changes() {
        let directory = std::env::temp_dir().join(format!(
            "constraintforge-resume-checkpoint-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(directory.join("trials")).unwrap();
        let trial = TrialRecord {
            trial_id: "trial-1".into(),
            workload: [1, 1, 1],
            budget: 1,
            policy_id: "random".into(),
            seed: 1,
            order_index: 0,
            status: "complete".into(),
            result_file: Some("trials/trial-1.json".into()),
            error: None,
        };
        fs::write(directory.join("trials/trial-1.json"), b"before").unwrap();
        let before = trial_hashes(&directory, std::slice::from_ref(&trial)).unwrap();
        durable_json(&directory.join("checkpoint.json"), &before).unwrap();
        assert!(durable_json(&directory.join("checkpoint.json"), &before).is_err());
        assert!(trial_hashes(&directory, &[trial.clone(), trial.clone()]).is_err());

        fs::write(directory.join("trials/trial-1.json"), b"after").unwrap();
        let after = trial_hashes(&directory, &[trial]).unwrap();
        assert_ne!(before[0].sha256, after[0].sha256);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn v2_validation_rejects_legacy_unequal_measurement_effort() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let protocol = Protocol::parse(&root.join("research/protocols/m6-v2-pilot.toml")).unwrap();
        let path = root.join("results/m6-pilot/trials/s01-m2k2n2-b2-random-seed11.json");
        let mut result: TuningResult = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        result.search_design = Some(SearchDesign::MatchedBatchesV2);
        result.software.research_implementation_version = Some("m6-v2".into());
        result
            .independent_evaluation_samples_ns
            .resize(protocol.evaluation_runs as usize, 1.0);
        let trial = TrialRecord {
            trial_id: result.trial_id.clone(),
            workload: result.shape_mkn,
            budget: 2,
            policy_id: "random".into(),
            seed: result.search_seed,
            order_index: 0,
            status: "complete".into(),
            result_file: Some("legacy.json".into()),
            error: None,
        };
        assert_eq!(result.search.measurement_batches, 2);
        assert_eq!(result.search.total_measured_dispatches, 4);
        assert!(
            validate_trial_result(&protocol, &trial, &result)
                .unwrap_err()
                .contains("unequal")
        );
    }
}

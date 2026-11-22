use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    time::Instant,
};

use constraintforge_bench::{SoftwareMetadata, current_software_metadata};
use constraintforge_tuner::{
    AdapterSummaryRecord, GpuReferenceBatchConfig, ReferenceBlockResult, run_gpu_reference_batch,
    tiled_candidate_ids,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ResearchError, atomic_json, now, write_new};

pub const REFERENCE_PANEL_PROTOCOL_SCHEMA: &str = "constraintforge-m8-phase2a-protocol-v1";
pub const REFERENCE_PANEL_PLAN_SCHEMA: &str = "constraintforge-m8-phase2a-plan-v1";
pub const REFERENCE_PANEL_SESSION_SCHEMA: &str = "constraintforge-m8-phase2a-session-v1";
pub const REFERENCE_PANEL_RECORD_SCHEMA: &str = "constraintforge-m8-phase2a-candidate-panel-v1";
pub const REFERENCE_PANEL_ANALYSIS_SCHEMA: &str = "constraintforge-m8-phase2a-analysis-v1";
pub const REFERENCE_PANEL_IMPLEMENTATION_VERSION: &str = "m8-phase2a-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReferencePanelProtocol {
    pub schema: String,
    pub id: String,
    pub version: String,
    pub implementation_version: String,
    pub phase: String,
    pub frozen: bool,
    pub classification: String,
    pub research_question: String,
    pub independent_unit: String,
    pub hardware_target: String,
    pub measurement: String,
    pub candidate_space: String,
    pub panel_definition: String,
    pub panel_ordering: String,
    pub candidate_ordering: String,
    pub block_ordering: String,
    pub correctness_policy: String,
    pub reference_aggregation: String,
    pub tie_rule: String,
    pub primary_metrics: Vec<String>,
    pub secondary_metrics: Vec<String>,
    pub invalid_rule: String,
    pub missing_rule: String,
    pub interruption_rule: String,
    pub stop_rules: String,
    pub workloads: Vec<[usize; 3]>,
    pub panel_ids: Vec<String>,
    pub candidates: Vec<String>,
    pub sessions: usize,
    pub seeds: Vec<u64>,
    pub blocks_per_candidate: usize,
    pub warmups_per_block: u32,
    pub samples_per_block: u32,
    pub correctness_admissions_per_block: u32,
    pub session_wall_cap_ms: u64,
    pub expected_panel_records: usize,
    pub expected_candidate_records: usize,
    pub expected_blocks: usize,
    pub expected_timed_samples: usize,
    pub expected_warmups: usize,
    pub expected_correctness_admissions: usize,
    pub expected_total_candidate_invocations: usize,
}

impl ReferencePanelProtocol {
    pub fn parse(path: &Path) -> Result<Self, ResearchError> {
        let protocol: Self = toml::from_str(&fs::read_to_string(path)?)?;
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ResearchError> {
        let candidates = tiled_candidate_ids();
        let panel_records = self.sessions * self.workloads.len() * self.panel_ids.len();
        let candidate_records = panel_records * candidates.len();
        let blocks = candidate_records * self.blocks_per_candidate;
        let samples = blocks * self.samples_per_block as usize;
        let warmups = blocks * self.warmups_per_block as usize;
        let correctness = blocks * self.correctness_admissions_per_block as usize;
        if self.schema != REFERENCE_PANEL_PROTOCOL_SCHEMA
            || self.implementation_version != REFERENCE_PANEL_IMPLEMENTATION_VERSION
            || self.phase != "frozen"
            || !self.frozen
            || self.classification != "PROSPECTIVE / DESCRIPTIVE PILOT"
            || self.workloads != [[7, 19, 11], [256, 64, 384]]
            || self.panel_ids != ["A", "B"]
            || self.candidates != candidates
            || self.sessions != 5
            || self.seeds.len() != 5
            || self.seeds.iter().collect::<BTreeSet<_>>().len() != 5
            || self.blocks_per_candidate != 3
            || self.warmups_per_block != 3
            || self.samples_per_block != 10
            || self.correctness_admissions_per_block != 1
            || self.expected_panel_records != panel_records
            || self.expected_candidate_records != candidate_records
            || self.expected_blocks != blocks
            || self.expected_timed_samples != samples
            || self.expected_warmups != warmups
            || self.expected_correctness_admissions != correctness
            || self.expected_total_candidate_invocations != samples + warmups + correctness
        {
            return Err(ResearchError::Invalid(
                "M8 Phase 2A protocol differs from the frozen bounded pilot",
            ));
        }
        if self.primary_metrics.is_empty() || self.secondary_metrics.is_empty() {
            return Err(ResearchError::Invalid("reference-panel metrics are empty"));
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
pub struct PlannedPanelBlock {
    pub block_index: usize,
    pub execution_order: usize,
    pub permutation_sha256: String,
    pub candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedPanel {
    pub workload: [usize; 3],
    pub workload_order: usize,
    pub panel_id: String,
    pub chronological_panel_order: usize,
    pub stream_seed_sha256: String,
    pub blocks: Vec<PlannedPanelBlock>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferencePanelPlan {
    pub schema: String,
    pub session_index: usize,
    pub session_seed: u64,
    pub panels: Vec<PlannedPanel>,
    pub expected_candidate_records: usize,
    pub expected_blocks: usize,
    pub expected_timed_samples: usize,
    pub expected_warmups: usize,
    pub expected_correctness_admissions: usize,
    pub expected_total_candidate_invocations: usize,
}

fn seeded_order(values: &[String], key: &str) -> (Vec<String>, String) {
    let stream = format!("{:x}", Sha256::digest(key.as_bytes()));
    let mut keyed = values
        .iter()
        .map(|value| {
            (
                Sha256::digest(format!("{key}:{value}").as_bytes()).to_vec(),
                value.clone(),
            )
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    (keyed.into_iter().map(|(_, value)| value).collect(), stream)
}

pub fn reference_panel_plan(
    protocol: &ReferencePanelProtocol,
    session_index: usize,
) -> Result<ReferencePanelPlan, ResearchError> {
    protocol.validate()?;
    if session_index >= protocol.sessions {
        return Err(ResearchError::Invalid("session index exceeds protocol"));
    }
    let seed = protocol.seeds[session_index];
    let mut workloads = protocol.workloads.clone();
    if session_index % 2 == 1 {
        workloads.reverse();
    }
    let mut panel_ids = protocol.panel_ids.clone();
    if session_index % 2 == 1 {
        panel_ids.reverse();
    }
    let mut panels = Vec::new();
    for (workload_order, workload) in workloads.into_iter().enumerate() {
        for panel_id in &panel_ids {
            let stream_key = format!(
                "m8-phase2a:{seed}:{}:{}:{}:{}",
                workload[0], workload[1], workload[2], panel_id
            );
            let (_, stream_seed_sha256) = seeded_order(&protocol.candidates, &stream_key);
            let block_names = (0..protocol.blocks_per_candidate)
                .map(|index| index.to_string())
                .collect::<Vec<_>>();
            let (block_order, _) = seeded_order(&block_names, &format!("{stream_key}:blocks"));
            let mut blocks = Vec::new();
            for (execution_order, block) in block_order.iter().enumerate() {
                let block_index = block
                    .parse::<usize>()
                    .map_err(|_| ResearchError::Invalid("invalid planned block index"))?;
                let (candidate_ids, permutation_sha256) = seeded_order(
                    &protocol.candidates,
                    &format!("{stream_key}:block:{block_index}"),
                );
                blocks.push(PlannedPanelBlock {
                    block_index,
                    execution_order,
                    permutation_sha256,
                    candidate_ids,
                });
            }
            panels.push(PlannedPanel {
                workload,
                workload_order,
                panel_id: panel_id.clone(),
                chronological_panel_order: panels.len(),
                stream_seed_sha256,
                blocks,
            });
        }
    }
    Ok(ReferencePanelPlan {
        schema: REFERENCE_PANEL_PLAN_SCHEMA.into(),
        session_index,
        session_seed: seed,
        panels,
        expected_candidate_records: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len(),
        expected_blocks: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len()
            * protocol.blocks_per_candidate,
        expected_timed_samples: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len()
            * protocol.blocks_per_candidate
            * protocol.samples_per_block as usize,
        expected_warmups: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len()
            * protocol.blocks_per_candidate
            * protocol.warmups_per_block as usize,
        expected_correctness_admissions: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len()
            * protocol.blocks_per_candidate
            * protocol.correctness_admissions_per_block as usize,
        expected_total_candidate_invocations: protocol.workloads.len()
            * protocol.panel_ids.len()
            * protocol.candidates.len()
            * protocol.blocks_per_candidate
            * (protocol.samples_per_block as usize
                + protocol.warmups_per_block as usize
                + protocol.correctness_admissions_per_block as usize),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PanelEnvironment {
    pub captured_utc: u64,
    pub power_source: Option<String>,
    pub battery_percent: Option<String>,
    pub thermal_state: Option<String>,
    pub background_load: Option<String>,
}

fn environment() -> Result<PanelEnvironment, ResearchError> {
    Ok(PanelEnvironment {
        captured_utc: now()?,
        power_source: std::env::var("CONSTRAINTFORGE_POWER_SOURCE").ok(),
        battery_percent: std::env::var("CONSTRAINTFORGE_BATTERY_PERCENT").ok(),
        thermal_state: std::env::var("CONSTRAINTFORGE_THERMAL_STATE").ok(),
        background_load: std::env::var("CONSTRAINTFORGE_BACKGROUND_LOAD").ok(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PanelAttempt {
    pub panel_id: String,
    pub workload: [usize; 3],
    pub candidate_id: String,
    pub block_index: usize,
    pub status: String,
    pub started_utc: u64,
    pub ended_utc: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidatePanelRecord {
    pub schema: String,
    pub synthetic: bool,
    pub session_id: String,
    pub session_index: usize,
    pub protocol_sha256: String,
    pub source_revision: String,
    pub source_tree_sha256: Option<String>,
    pub panel_id: String,
    pub chronological_panel_order: usize,
    pub workload: [usize; 3],
    pub candidate_id: String,
    pub adapter: Option<AdapterSummaryRecord>,
    pub correctness_admission_blocks: Vec<usize>,
    pub blocks: Vec<ReferenceBlockResult>,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferencePanelSession {
    pub schema: String,
    pub synthetic: bool,
    pub session_id: String,
    pub session_index: usize,
    pub protocol_sha256: String,
    pub software: SoftwareMetadata,
    pub plan: ReferencePanelPlan,
    pub phase: String,
    pub record_files: Vec<String>,
    pub attempts: Vec<PanelAttempt>,
    pub environment: Vec<PanelEnvironment>,
    pub source_preflight_captured_utc: u64,
    pub generated_output_created_after_preflight: bool,
    pub started_utc: u64,
    pub ended_utc: Option<u64>,
}

type PanelRunResult = (
    AdapterSummaryRecord,
    SoftwareMetadata,
    ReferenceBlockResult,
    u64,
    u64,
);

fn record_path(panel: &PlannedPanel, candidate: &str) -> String {
    format!(
        "panels/m{}k{}n{}/{}/{}.json",
        panel.workload[0], panel.workload[1], panel.workload[2], panel.panel_id, candidate
    )
}

fn validate_output_path(path: &Path) -> Result<(), ResearchError> {
    if path.as_os_str().is_empty()
        || path.file_name().is_none()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ResearchError::Invalid(
            "reference-panel output must be a safe relative path",
        ));
    }
    Ok(())
}

fn initialize_storage(
    output: &Path,
    plan: &ReferencePanelPlan,
    candidates: &[String],
) -> Result<(), ResearchError> {
    let mut parents = BTreeSet::new();
    for panel in &plan.panels {
        for candidate in candidates {
            let path = output.join(record_path(panel, candidate));
            let parent = path
                .parent()
                .ok_or(ResearchError::Invalid("record path has no parent"))?;
            fs::create_dir_all(parent)?;
            parents.insert(parent.to_path_buf());
        }
    }
    for parent in parents {
        let probe = parent.join(format!(".persistence-probe-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)?;
        file.write_all(b"reference-panel-storage-ready\n")?;
        file.sync_all()?;
        fs::remove_file(probe)?;
    }
    Ok(())
}

pub fn run_reference_panel_session(
    protocol: &ReferencePanelProtocol,
    session_id: &str,
    session_index: usize,
    output: &Path,
    resume: bool,
) -> Result<ReferencePanelSession, ResearchError> {
    run_reference_panel_session_with(
        protocol,
        session_id,
        session_index,
        output,
        resume,
        false,
        || current_software_metadata().map_err(ResearchError::from),
        |_, config| run_gpu_reference_batch(config).map_err(|error| error.to_string()),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_reference_panel_session_with<Metadata, Measure>(
    protocol: &ReferencePanelProtocol,
    session_id: &str,
    session_index: usize,
    output: &Path,
    resume: bool,
    synthetic: bool,
    mut metadata: Metadata,
    mut measure: Measure,
) -> Result<ReferencePanelSession, ResearchError>
where
    Metadata: FnMut() -> Result<SoftwareMetadata, ResearchError>,
    Measure: FnMut(&PlannedPanel, &GpuReferenceBatchConfig) -> Result<PanelRunResult, String>,
{
    protocol.validate()?;
    validate_output_path(output)?;
    let plan = reference_panel_plan(protocol, session_index)?;
    let protocol_sha256 = protocol.sha256()?;
    let captured = now()?;
    let mut software = metadata()?;
    software.research_implementation_version = Some(REFERENCE_PANEL_IMPLEMENTATION_VERSION.into());
    if !synthetic && software.git_worktree_clean != Some(true) {
        return Err(ResearchError::Invalid(
            "real reference-panel execution requires a committed clean source tree",
        ));
    }
    if output.exists() && !resume {
        return Err(ResearchError::OutputExists(output.into()));
    }
    if resume && !output.exists() {
        return Err(ResearchError::Invalid("cannot resume a missing session"));
    }
    fs::create_dir_all(output)?;
    let protocol_path = output.join("resolved-protocol.toml");
    if !protocol_path.exists() {
        write_new(&protocol_path, toml::to_string_pretty(protocol)?.as_bytes())?;
    }
    let state_path = output.join("session.json");
    let mut state = if state_path.exists() {
        let state: ReferencePanelSession = serde_json::from_slice(&fs::read(&state_path)?)?;
        let saved = ReferencePanelProtocol::parse(&protocol_path)?;
        if !resume
            || state.session_id != session_id
            || state.session_index != session_index
            || state.protocol_sha256 != protocol_sha256
            || state.plan != plan
            || saved.sha256()? != protocol_sha256
            || state.software.source_revision != software.source_revision
            || state.software.source_tree_sha256 != software.source_tree_sha256
        {
            return Err(ResearchError::Invalid(
                "reference-panel resume identity mismatch",
            ));
        }
        state
    } else {
        ReferencePanelSession {
            schema: REFERENCE_PANEL_SESSION_SCHEMA.into(),
            synthetic,
            session_id: session_id.into(),
            session_index,
            protocol_sha256: protocol_sha256.clone(),
            software,
            plan: plan.clone(),
            phase: "measurement".into(),
            record_files: Vec::new(),
            attempts: Vec::new(),
            environment: vec![environment()?],
            source_preflight_captured_utc: captured,
            generated_output_created_after_preflight: true,
            started_utc: now()?,
            ended_utc: None,
        }
    };
    atomic_json(&state_path, &state)?;
    initialize_storage(output, &plan, &protocol.candidates)?;
    reconcile_attempts(output, &mut state)?;
    atomic_json(&state_path, &state)?;
    let session_started = Instant::now();

    for panel in &plan.panels {
        for block in &panel.blocks {
            for (order_index, candidate) in block.candidate_ids.iter().enumerate() {
                if session_started.elapsed().as_millis() >= protocol.session_wall_cap_ms as u128 {
                    return Err(ResearchError::Invalid(
                        "reference-panel session wall-clock cap reached",
                    ));
                }
                let relative = record_path(panel, candidate);
                let path = output.join(&relative);
                let mut record = if path.exists() {
                    serde_json::from_slice::<CandidatePanelRecord>(&fs::read(&path)?)?
                } else {
                    CandidatePanelRecord {
                        schema: REFERENCE_PANEL_RECORD_SCHEMA.into(),
                        synthetic,
                        session_id: session_id.into(),
                        session_index,
                        protocol_sha256: protocol_sha256.clone(),
                        source_revision: state.software.source_revision.clone(),
                        source_tree_sha256: state.software.source_tree_sha256.clone(),
                        panel_id: panel.panel_id.clone(),
                        chronological_panel_order: panel.chronological_panel_order,
                        workload: panel.workload,
                        candidate_id: candidate.clone(),
                        adapter: None,
                        correctness_admission_blocks: Vec::new(),
                        blocks: Vec::new(),
                        status: "incomplete".into(),
                    }
                };
                validate_record_identity(&record, &state, panel, candidate)?;
                if record
                    .blocks
                    .iter()
                    .any(|saved| saved.block_index == block.block_index)
                {
                    continue;
                }
                if record.status == "complete" {
                    return Err(ResearchError::Invalid(
                        "complete candidate record is missing a planned block",
                    ));
                }
                state.attempts.push(PanelAttempt {
                    panel_id: panel.panel_id.clone(),
                    workload: panel.workload,
                    candidate_id: candidate.clone(),
                    block_index: block.block_index,
                    status: "started_or_possibly_executed".into(),
                    started_utc: now()?,
                    ended_utc: None,
                });
                atomic_json(&state_path, &state)?;
                let (adapter, measured_software, measured, _, _) = measure(
                    panel,
                    &GpuReferenceBatchConfig {
                        shape: panel.workload,
                        workload_seed: plan.session_seed,
                        candidate_id: candidate.clone(),
                        block_index: block.block_index,
                        order_index,
                        warmups: protocol.warmups_per_block,
                        samples: protocol.samples_per_block,
                    },
                )
                .map_err(|_| ResearchError::Invalid("reference-panel measurement failed"))?;
                if measured_software.source_revision != state.software.source_revision
                    || measured_software.source_tree_sha256 != state.software.source_tree_sha256
                {
                    return Err(ResearchError::Invalid(
                        "source identity changed during reference-panel execution",
                    ));
                }
                validate_adapter(&adapter)?;
                validate_block(protocol, block, candidate, &measured)?;
                if record.adapter.as_ref().is_some_and(|saved| {
                    saved.name != adapter.name
                        || saved.backend != adapter.backend
                        || saved.timestamp_period_ns != adapter.timestamp_period_ns
                }) {
                    return Err(ResearchError::Invalid(
                        "adapter identity changed within candidate panel",
                    ));
                }
                record.adapter = Some(adapter);
                record.correctness_admission_blocks.push(block.block_index);
                record.blocks.push(measured);
                record.blocks.sort_by_key(|saved| saved.block_index);
                record.correctness_admission_blocks.sort_unstable();
                if record.blocks.len() == protocol.blocks_per_candidate {
                    record.status = "complete".into();
                }
                atomic_json(&path, &record)?;
                let attempt = state
                    .attempts
                    .iter_mut()
                    .find(|attempt| {
                        attempt.status == "started_or_possibly_executed"
                            && attempt.panel_id == panel.panel_id
                            && attempt.workload == panel.workload
                            && attempt.candidate_id == *candidate
                            && attempt.block_index == block.block_index
                    })
                    .ok_or(ResearchError::Invalid("durable attempt disappeared"))?;
                attempt.status = "completed_and_durably_recorded".into();
                attempt.ended_utc = Some(now()?);
                if record.status == "complete" && !state.record_files.contains(&relative) {
                    state.record_files.push(relative);
                    state.record_files.sort();
                }
                atomic_json(&state_path, &state)?;
            }
        }
    }
    validate_session(protocol, &state, output)?;
    state.phase = "complete".into();
    state.environment.push(environment()?);
    state.ended_utc = Some(now()?);
    atomic_json(&state_path, &state)?;
    Ok(state)
}

fn validate_adapter(adapter: &AdapterSummaryRecord) -> Result<(), ResearchError> {
    if adapter.name != "Apple M5"
        || adapter.backend != "Metal"
        || !adapter.timestamp_query_supported
        || !adapter
            .timestamp_period_ns
            .is_some_and(|value| value.is_finite() && value > 0.0)
    {
        return Err(ResearchError::Invalid(
            "reference-panel hardware or timestamp requirement failed",
        ));
    }
    Ok(())
}

fn validate_block(
    protocol: &ReferencePanelProtocol,
    planned: &PlannedPanelBlock,
    candidate: &str,
    block: &ReferenceBlockResult,
) -> Result<(), ResearchError> {
    if block.block_index != planned.block_index
        || block.candidate_id != candidate
        || block.raw_samples_ns.len() != protocol.samples_per_block as usize
        || block
            .raw_samples_ns
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ResearchError::Invalid("invalid reference-panel block"));
    }
    Ok(())
}

fn validate_record_identity(
    record: &CandidatePanelRecord,
    state: &ReferencePanelSession,
    panel: &PlannedPanel,
    candidate: &str,
) -> Result<(), ResearchError> {
    if record.schema != REFERENCE_PANEL_RECORD_SCHEMA
        || record.synthetic != state.synthetic
        || record.session_id != state.session_id
        || record.session_index != state.session_index
        || record.protocol_sha256 != state.protocol_sha256
        || record.source_revision != state.software.source_revision
        || record.source_tree_sha256 != state.software.source_tree_sha256
        || record.panel_id != panel.panel_id
        || record.chronological_panel_order != panel.chronological_panel_order
        || record.workload != panel.workload
        || record.candidate_id != candidate
    {
        return Err(ResearchError::Invalid("candidate-panel identity mismatch"));
    }
    Ok(())
}

fn reconcile_attempts(
    output: &Path,
    state: &mut ReferencePanelSession,
) -> Result<(), ResearchError> {
    for attempt in &mut state.attempts {
        if attempt.status != "started_or_possibly_executed" {
            continue;
        }
        let panel = state
            .plan
            .panels
            .iter()
            .find(|panel| panel.panel_id == attempt.panel_id && panel.workload == attempt.workload)
            .ok_or(ResearchError::Invalid("attempt panel is not planned"))?;
        let path = output.join(record_path(panel, &attempt.candidate_id));
        if path.exists() {
            let record: CandidatePanelRecord = serde_json::from_slice(&fs::read(path)?)?;
            if record
                .blocks
                .iter()
                .any(|block| block.block_index == attempt.block_index)
            {
                attempt.status = "completed_and_durably_recorded".into();
                attempt.ended_utc = Some(now()?);
                continue;
            }
        }
        return Err(ResearchError::Invalid(
            "ambiguous reference-panel invocation cannot be remeasured",
        ));
    }
    Ok(())
}

fn validate_session(
    protocol: &ReferencePanelProtocol,
    state: &ReferencePanelSession,
    root: &Path,
) -> Result<(), ResearchError> {
    if state.schema != REFERENCE_PANEL_SESSION_SCHEMA
        || state.protocol_sha256 != protocol.sha256()?
        || state.plan != reference_panel_plan(protocol, state.session_index)?
        || state.record_files.len() != state.plan.expected_candidate_records
        || state.attempts.len() != state.plan.expected_blocks
        || state
            .attempts
            .iter()
            .any(|attempt| attempt.status != "completed_and_durably_recorded")
    {
        return Err(ResearchError::Invalid("incomplete reference-panel session"));
    }
    let mut attempts = BTreeSet::new();
    for attempt in &state.attempts {
        let panel = state
            .plan
            .panels
            .iter()
            .find(|panel| panel.panel_id == attempt.panel_id && panel.workload == attempt.workload)
            .ok_or(ResearchError::Invalid("attempt panel is not planned"))?;
        let block = panel
            .blocks
            .iter()
            .find(|block| block.block_index == attempt.block_index)
            .ok_or(ResearchError::Invalid("attempt block is not planned"))?;
        if !block.candidate_ids.contains(&attempt.candidate_id)
            || !attempts.insert((
                attempt.workload,
                attempt.panel_id.clone(),
                attempt.candidate_id.clone(),
                attempt.block_index,
            ))
        {
            return Err(ResearchError::Invalid("duplicate or unplanned attempt"));
        }
    }
    let mut identities = BTreeSet::new();
    for relative in &state.record_files {
        let record: CandidatePanelRecord = serde_json::from_slice(&fs::read(root.join(relative))?)?;
        let panel = state
            .plan
            .panels
            .iter()
            .find(|panel| panel.panel_id == record.panel_id && panel.workload == record.workload)
            .ok_or(ResearchError::Invalid("record panel is not planned"))?;
        validate_record_identity(&record, state, panel, &record.candidate_id)?;
        if !identities.insert((
            record.workload,
            record.panel_id.clone(),
            record.candidate_id.clone(),
        )) || record.status != "complete"
            || record.blocks.len() != protocol.blocks_per_candidate
            || record.correctness_admission_blocks.len() != protocol.blocks_per_candidate
            || record.correctness_admission_blocks
                != (0..protocol.blocks_per_candidate).collect::<Vec<_>>()
        {
            return Err(ResearchError::Invalid(
                "duplicate or incomplete candidate panel",
            ));
        }
        for block in &record.blocks {
            let planned = panel
                .blocks
                .iter()
                .find(|planned| planned.block_index == block.block_index)
                .ok_or(ResearchError::Invalid("record block is not planned"))?;
            validate_block(protocol, planned, &record.candidate_id, block)?;
        }
        validate_adapter(record.adapter.as_ref().ok_or(ResearchError::Invalid(
            "candidate panel lacks adapter provenance",
        ))?)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferencePanelValidation {
    pub schema: String,
    pub valid: bool,
    pub sessions: usize,
    pub candidate_records: usize,
    pub blocks: usize,
    pub timed_samples: usize,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MeasurementSourceIdentity {
    source_revision: String,
    source_tree_sha256: String,
    source_manifest_sha256: Vec<constraintforge_bench::SourceFileHash>,
    cargo_lock_sha256: String,
    rust_toolchain_sha256: String,
    implementation_version: String,
}

fn measurement_source_identity(
    session_id: &str,
    software: &SoftwareMetadata,
) -> Result<MeasurementSourceIdentity, String> {
    let required = |field: &str, value: Option<&str>| {
        value
            .filter(|value| !value.is_empty() && *value != "unavailable/unversioned")
            .map(str::to_owned)
            .ok_or_else(|| format!("session {session_id} is missing measurement source {field}"))
    };
    let source_revision = required("revision", Some(&software.source_revision))?;
    let source_tree_sha256 = required("tree SHA-256", software.source_tree_sha256.as_deref())?;
    let cargo_lock_sha256 = required("Cargo.lock SHA-256", software.cargo_lock_sha256.as_deref())?;
    let rust_toolchain_sha256 = required(
        "rust-toolchain SHA-256",
        software.rust_toolchain_sha256.as_deref(),
    )?;
    let implementation_version = required(
        "implementation version",
        software.research_implementation_version.as_deref(),
    )?;
    if implementation_version != REFERENCE_PANEL_IMPLEMENTATION_VERSION {
        return Err(format!(
            "session {session_id} has conflicting measurement source implementation version"
        ));
    }
    if software.source_manifest_sha256.is_empty() {
        return Err(format!(
            "session {session_id} is missing measurement source per-file manifest"
        ));
    }
    let mut tree = Sha256::new();
    let mut previous = None;
    for entry in &software.source_manifest_sha256 {
        if entry.path.is_empty()
            || entry.sha256.len() != 64
            || !entry.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || previous.is_some_and(|path| path >= entry.path.as_str())
        {
            return Err(format!(
                "session {session_id} has invalid measurement source per-file manifest"
            ));
        }
        tree.update(entry.path.as_bytes());
        tree.update([0]);
        tree.update(entry.sha256.as_bytes());
        tree.update([0xff]);
        previous = Some(entry.path.as_str());
    }
    if format!("{:x}", tree.finalize()) != source_tree_sha256 {
        return Err(format!(
            "session {session_id} has a measurement source tree SHA-256 inconsistent with its per-file manifest"
        ));
    }
    Ok(MeasurementSourceIdentity {
        source_revision,
        source_tree_sha256,
        source_manifest_sha256: software.source_manifest_sha256.clone(),
        cargo_lock_sha256,
        rust_toolchain_sha256,
        implementation_version,
    })
}

fn validate_campaign_source_identities(
    identities: &[(usize, String, MeasurementSourceIdentity)],
) -> Result<(), String> {
    if identities.len() <= 1 {
        return Ok(());
    }
    let (_, authoritative_session, authoritative) = identities
        .iter()
        .find(|(index, _, _)| *index == 0)
        .ok_or_else(|| "multi-session campaign lacks authoritative Session index 0".to_owned())?;
    for (_, session_id, identity) in identities {
        for (field, matches) in [
            (
                "source_revision",
                identity.source_revision == authoritative.source_revision,
            ),
            (
                "source_tree_sha256",
                identity.source_tree_sha256 == authoritative.source_tree_sha256,
            ),
            (
                "source_manifest_sha256",
                identity.source_manifest_sha256 == authoritative.source_manifest_sha256,
            ),
            (
                "research_implementation_version",
                identity.implementation_version == authoritative.implementation_version,
            ),
            (
                "cargo_lock_sha256",
                identity.cargo_lock_sha256 == authoritative.cargo_lock_sha256,
            ),
            (
                "rust_toolchain_sha256",
                identity.rust_toolchain_sha256 == authoritative.rust_toolchain_sha256,
            ),
        ] {
            if !matches {
                return Err(format!(
                    "session {session_id} measurement source {field} conflicts with authoritative {authoritative_session}"
                ));
            }
        }
    }
    Ok(())
}

fn session_dirs(input: &Path) -> Result<Vec<PathBuf>, ResearchError> {
    if input.join("session.json").is_file() {
        return Ok(vec![input.into()]);
    }
    let mut dirs = fs::read_dir(input)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir() && path.join("session.json").is_file())
        .collect::<Vec<_>>();
    dirs.sort();
    Ok(dirs)
}

pub fn validate_reference_panel_results(
    input: &Path,
) -> Result<ReferencePanelValidation, ResearchError> {
    let mut validation = ReferencePanelValidation {
        schema: "constraintforge-m8-phase2a-validation-v1".into(),
        valid: true,
        sessions: 0,
        candidate_records: 0,
        blocks: 0,
        timed_samples: 0,
        errors: Vec::new(),
    };
    let dirs = session_dirs(input)?;
    let mut indices = BTreeSet::new();
    let mut source_identities = Vec::new();
    for dir in dirs {
        let result = (|| -> Result<_, String> {
            let protocol = ReferencePanelProtocol::parse(&dir.join("resolved-protocol.toml"))
                .map_err(|error| error.to_string())?;
            let state: ReferencePanelSession = serde_json::from_slice(
                &fs::read(dir.join("session.json")).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            if state.phase != "complete" || !indices.insert(state.session_index) {
                return Err("duplicate or incomplete session".into());
            }
            validate_session(&protocol, &state, &dir).map_err(|error| error.to_string())?;
            let identity = measurement_source_identity(&state.session_id, &state.software)?;
            Ok((
                state.session_index,
                state.session_id,
                identity,
                state.record_files.len(),
                state.plan.expected_blocks,
                state.plan.expected_timed_samples,
            ))
        })();
        match result {
            Ok((index, session_id, identity, records, blocks, samples)) => {
                validation.sessions += 1;
                validation.candidate_records += records;
                validation.blocks += blocks;
                validation.timed_samples += samples;
                source_identities.push((index, session_id, identity));
            }
            Err(error) => validation.errors.push(error),
        }
    }
    if let Err(error) = validate_campaign_source_identities(&source_identities) {
        validation.errors.push(error);
    }
    if validation.sessions == 0 {
        validation.errors.push("no complete sessions found".into());
    }
    validation.valid = validation.errors.is_empty();
    Ok(validation)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferencePanelAnalysis {
    pub schema: String,
    pub classification: String,
    pub sessions: usize,
    pub workload_pairs: usize,
    pub candidate_rows: usize,
    pub source_protocol_sha256: String,
}

#[derive(Clone)]
struct PanelSummary {
    session_id: String,
    session_index: usize,
    workload: [usize; 3],
    panel_id: String,
    latencies: BTreeMap<String, f64>,
    best: Vec<String>,
    ranks: BTreeMap<String, f64>,
    variability: BTreeMap<String, [f64; 4]>,
    best_runner_up_separation: f64,
}

fn median(values: &[f64]) -> Result<f64, ResearchError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ResearchError::Invalid("invalid latency sample"));
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    Ok(if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    })
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn ranks(values: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let mut ordered = values.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(b.0)));
    let mut result = BTreeMap::new();
    let mut index = 0;
    while index < ordered.len() {
        let mut end = index + 1;
        while end < ordered.len() && ordered[end].1 == ordered[index].1 {
            end += 1;
        }
        let rank = ((index + 1) + end) as f64 / 2.0;
        for (candidate, _) in &ordered[index..end] {
            result.insert((*candidate).clone(), rank);
        }
        index = end;
    }
    result
}

fn correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let scale = (left.iter().map(|x| (x - left_mean).powi(2)).sum::<f64>()
        * right.iter().map(|x| (x - right_mean).powi(2)).sum::<f64>())
    .sqrt();
    (scale > 0.0).then_some(covariance / scale)
}

fn kendall_tau_b(left: &[f64], right: &[f64]) -> Option<f64> {
    let mut concordant: f64 = 0.0;
    let mut discordant: f64 = 0.0;
    let mut left_ties: f64 = 0.0;
    let mut right_ties: f64 = 0.0;
    for i in 0..left.len() {
        for j in i + 1..left.len() {
            let a = left[i].total_cmp(&left[j]);
            let b = right[i].total_cmp(&right[j]);
            if a.is_eq() && !b.is_eq() {
                left_ties += 1.0;
            }
            if b.is_eq() && !a.is_eq() {
                right_ties += 1.0;
            }
            if a == b && !a.is_eq() {
                concordant += 1.0;
            } else if !a.is_eq() && !b.is_eq() {
                discordant += 1.0;
            }
        }
    }
    let denominator =
        ((concordant + discordant + left_ties) * (concordant + discordant + right_ties)).sqrt();
    if denominator > 0.0 {
        Some((concordant - discordant) / denominator)
    } else {
        None
    }
}

fn load_summaries(input: &Path) -> Result<(Vec<PanelSummary>, String), ResearchError> {
    let mut summaries = Vec::new();
    let mut protocol_hash = None;
    for dir in session_dirs(input)? {
        let protocol = ReferencePanelProtocol::parse(&dir.join("resolved-protocol.toml"))?;
        let hash = protocol.sha256()?;
        if protocol_hash.as_ref().is_some_and(|saved| saved != &hash) {
            return Err(ResearchError::Invalid("sessions use different protocols"));
        }
        protocol_hash = Some(hash);
        let state: ReferencePanelSession =
            serde_json::from_slice(&fs::read(dir.join("session.json"))?)?;
        validate_session(&protocol, &state, &dir)?;
        for workload in &protocol.workloads {
            for panel_id in &protocol.panel_ids {
                let mut latencies = BTreeMap::new();
                let mut variability = BTreeMap::new();
                for candidate in &protocol.candidates {
                    let panel = state
                        .plan
                        .panels
                        .iter()
                        .find(|panel| panel.workload == *workload && panel.panel_id == *panel_id)
                        .ok_or(ResearchError::Invalid("planned panel missing"))?;
                    let record: CandidatePanelRecord = serde_json::from_slice(&fs::read(
                        dir.join(record_path(panel, candidate)),
                    )?)?;
                    let samples = record
                        .blocks
                        .iter()
                        .flat_map(|block| block.raw_samples_ns.iter().copied())
                        .collect::<Vec<_>>();
                    latencies.insert(candidate.clone(), median(&samples)?);
                    let block_medians = record
                        .blocks
                        .iter()
                        .map(|block| median(&block.raw_samples_ns))
                        .collect::<Result<Vec<_>, _>>()?;
                    variability.insert(
                        candidate.clone(),
                        [
                            record
                                .blocks
                                .iter()
                                .map(|block| coefficient_of_variation(&block.raw_samples_ns))
                                .max_by(f64::total_cmp)
                                .unwrap(),
                            coefficient_of_variation(&block_medians),
                            samples.iter().copied().min_by(f64::total_cmp).unwrap(),
                            samples.iter().copied().max_by(f64::total_cmp).unwrap(),
                        ],
                    );
                }
                let minimum = latencies
                    .values()
                    .copied()
                    .min_by(f64::total_cmp)
                    .ok_or(ResearchError::Invalid("panel has no candidate latencies"))?;
                let best = latencies
                    .iter()
                    .filter(|(_, latency)| **latency == minimum)
                    .map(|(candidate, _)| candidate.clone())
                    .collect::<Vec<_>>();
                let runner_up = latencies
                    .values()
                    .copied()
                    .filter(|latency| *latency > minimum)
                    .min_by(f64::total_cmp);
                summaries.push(PanelSummary {
                    session_id: state.session_id.clone(),
                    session_index: state.session_index,
                    workload: *workload,
                    panel_id: panel_id.clone(),
                    ranks: ranks(&latencies),
                    latencies,
                    best,
                    variability,
                    best_runner_up_separation: runner_up
                        .map_or(0.0, |latency| (latency - minimum) / minimum),
                });
            }
        }
    }
    Ok((
        summaries,
        protocol_hash.ok_or(ResearchError::Invalid("no protocol"))?,
    ))
}

fn csv_value(value: Option<f64>) -> String {
    value.map_or_else(|| "undefined".into(), |value| value.to_string())
}

pub fn analyze_reference_panels(
    input: &Path,
    output: &Path,
    qualification_only: bool,
) -> Result<ReferencePanelAnalysis, ResearchError> {
    if output.exists() {
        return Err(ResearchError::OutputExists(output.into()));
    }
    let validation = validate_reference_panel_results(input)?;
    let expected = if qualification_only { 1 } else { 5 };
    if !validation.valid || validation.sessions != expected {
        return Err(ResearchError::Invalid(
            "reference-panel analysis requires the exact validated session count",
        ));
    }
    let (summaries, protocol_hash) = load_summaries(input)?;
    fs::create_dir_all(output)?;
    let mut panel_candidates = String::from(
        "schema,session_id,session_index,m,k,n,panel_id,candidate_id,latency_ns,rank,is_best\n",
    );
    for summary in &summaries {
        for (candidate, latency) in &summary.latencies {
            panel_candidates.push_str(&format!(
                "constraintforge-m8-phase2a-panel-candidate-v1,{},{},{},{},{},{},{},{},{},{}\n",
                summary.session_id,
                summary.session_index,
                summary.workload[0],
                summary.workload[1],
                summary.workload[2],
                summary.panel_id,
                candidate,
                latency,
                summary.ranks[candidate],
                summary.best.contains(candidate)
            ));
        }
    }
    let mut best_csv = String::from(
        "schema,session_id,session_index,m,k,n,best_a,best_b,exact_agreement,jaccard\n",
    );
    let mut rank_csv =
        String::from("schema,session_id,session_index,m,k,n,spearman,kendall_tau_b,top3_jaccard\n");
    let mut regret_csv = String::from(
        "schema,session_id,session_index,m,k,n,a_to_b_min,a_to_b_max,a_to_b_designated,b_to_a_min,b_to_a_max,b_to_a_designated\n",
    );
    let mut variability_csv = String::from(
        "schema,session_id,session_index,m,k,n,candidate_id,latency_a_ns,latency_b_ns,ratio_a_b,relative_difference,a_best_runner_up_separation,b_best_runner_up_separation,a_max_within_block_cv,b_max_within_block_cv,a_between_block_median_cv,b_between_block_median_cv,a_min_sample_ns,a_max_sample_ns,b_min_sample_ns,b_max_sample_ns\n",
    );
    let mut ranking_csv = String::from(
        "schema,session_id,session_index,m,k,n,candidate_id,rank_a,rank_b,rank_change\n",
    );
    let mut pairs = 0;
    for a in summaries.iter().filter(|summary| summary.panel_id == "A") {
        let b = summaries
            .iter()
            .find(|summary| {
                summary.panel_id == "B"
                    && summary.session_index == a.session_index
                    && summary.workload == a.workload
            })
            .ok_or(ResearchError::Invalid("paired panel B missing"))?;
        pairs += 1;
        let set_a = a.best.iter().cloned().collect::<BTreeSet<_>>();
        let set_b = b.best.iter().cloned().collect::<BTreeSet<_>>();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        best_csv.push_str(&format!(
            "constraintforge-m8-phase2a-best-set-v1,{},{},{},{},{},{},{},{},{}\n",
            a.session_id,
            a.session_index,
            a.workload[0],
            a.workload[1],
            a.workload[2],
            a.best.join("|"),
            b.best.join("|"),
            set_a == set_b,
            intersection as f64 / union as f64
        ));
        let candidates = a.latencies.keys().cloned().collect::<Vec<_>>();
        if candidates
            .iter()
            .any(|candidate| !b.latencies.contains_key(candidate))
        {
            return Err(ResearchError::Invalid(
                "panels use different candidate sets",
            ));
        }
        let rank_a = candidates
            .iter()
            .map(|candidate| a.ranks[candidate])
            .collect::<Vec<_>>();
        let rank_b = candidates
            .iter()
            .map(|candidate| b.ranks[candidate])
            .collect::<Vec<_>>();
        let top_a = candidates
            .iter()
            .filter(|candidate| a.ranks[*candidate] <= 3.0)
            .cloned()
            .collect::<BTreeSet<_>>();
        let top_b = candidates
            .iter()
            .filter(|candidate| b.ranks[*candidate] <= 3.0)
            .cloned()
            .collect::<BTreeSet<_>>();
        let top_union = top_a.union(&top_b).count();
        let top_jaccard = if top_union == 0 {
            None
        } else {
            Some(top_a.intersection(&top_b).count() as f64 / top_union as f64)
        };
        rank_csv.push_str(&format!(
            "constraintforge-m8-phase2a-rank-agreement-v1,{},{},{},{},{},{},{},{}\n",
            a.session_id,
            a.session_index,
            a.workload[0],
            a.workload[1],
            a.workload[2],
            csv_value(correlation(&rank_a, &rank_b)),
            csv_value(kendall_tau_b(&rank_a, &rank_b)),
            csv_value(top_jaccard)
        ));
        let min_a = a
            .latencies
            .values()
            .copied()
            .min_by(f64::total_cmp)
            .unwrap();
        let min_b = b
            .latencies
            .values()
            .copied()
            .min_by(f64::total_cmp)
            .unwrap();
        let regrets_ab = a
            .best
            .iter()
            .map(|candidate| (b.latencies[candidate] - min_b) / min_b)
            .collect::<Vec<_>>();
        let regrets_ba = b
            .best
            .iter()
            .map(|candidate| (a.latencies[candidate] - min_a) / min_a)
            .collect::<Vec<_>>();
        regret_csv.push_str(&format!(
            "constraintforge-m8-phase2a-cross-panel-regret-v1,{},{},{},{},{},{},{},{},{},{},{}\n",
            a.session_id,
            a.session_index,
            a.workload[0],
            a.workload[1],
            a.workload[2],
            regrets_ab.iter().copied().min_by(f64::total_cmp).unwrap(),
            regrets_ab.iter().copied().max_by(f64::total_cmp).unwrap(),
            regrets_ab[0],
            regrets_ba.iter().copied().min_by(f64::total_cmp).unwrap(),
            regrets_ba.iter().copied().max_by(f64::total_cmp).unwrap(),
            regrets_ba[0]
        ));
        for candidate in candidates {
            let latency_a = a.latencies[&candidate];
            let latency_b = b.latencies[&candidate];
            variability_csv.push_str(&format!(
                "constraintforge-m8-phase2a-variability-v1,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                a.session_id,
                a.session_index,
                a.workload[0],
                a.workload[1],
                a.workload[2],
                candidate,
                latency_a,
                latency_b,
                latency_a / latency_b,
                (latency_a - latency_b) / latency_b,
                a.best_runner_up_separation,
                b.best_runner_up_separation,
                a.variability[&candidate][0],
                b.variability[&candidate][0],
                a.variability[&candidate][1],
                b.variability[&candidate][1],
                a.variability[&candidate][2],
                a.variability[&candidate][3],
                b.variability[&candidate][2],
                b.variability[&candidate][3]
            ));
            ranking_csv.push_str(&format!(
                "constraintforge-m8-phase2a-ranking-v1,{},{},{},{},{},{},{},{},{}\n",
                a.session_id,
                a.session_index,
                a.workload[0],
                a.workload[1],
                a.workload[2],
                candidate,
                a.ranks[&candidate],
                b.ranks[&candidate],
                a.ranks[&candidate] - b.ranks[&candidate]
            ));
        }
    }
    let analysis = ReferencePanelAnalysis {
        schema: REFERENCE_PANEL_ANALYSIS_SCHEMA.into(),
        classification: if qualification_only {
            "SINGLE-SESSION QUALIFICATION / DESCRIPTIVE ONLY".into()
        } else {
            "PROSPECTIVE / DESCRIPTIVE PILOT".into()
        },
        sessions: validation.sessions,
        workload_pairs: pairs,
        candidate_rows: summaries.len() * 10,
        source_protocol_sha256: protocol_hash,
    };
    write_new(
        output.join("panel-candidates.csv").as_path(),
        panel_candidates.as_bytes(),
    )?;
    write_new(
        output.join("panel-rankings.csv").as_path(),
        ranking_csv.as_bytes(),
    )?;
    write_new(
        output.join("best-set-agreement.csv").as_path(),
        best_csv.as_bytes(),
    )?;
    write_new(
        output.join("rank-agreement.csv").as_path(),
        rank_csv.as_bytes(),
    )?;
    write_new(
        output.join("cross-panel-regret.csv").as_path(),
        regret_csv.as_bytes(),
    )?;
    write_new(
        output.join("reference-variability.csv").as_path(),
        variability_csv.as_bytes(),
    )?;
    write_new(
        output.join("validation.json").as_path(),
        &serde_json::to_vec_pretty(&validation)?,
    )?;
    write_new(
        output.join("session-summary.json").as_path(),
        &serde_json::to_vec_pretty(&analysis)?,
    )?;
    write_new(
        output.join("resolved-protocol.json").as_path(),
        &serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "constraintforge-m8-phase2a-resolved-analysis-v1",
            "protocol_sha256": analysis.source_protocol_sha256,
            "sessions": analysis.sessions,
        }))?,
    )?;
    write_new(
        output.join("summary.md").as_path(),
        format!(
            "# M8 Phase 2A reference-panel analysis\n\nClassification: **{}**.\n\nSessions are the independent unit; panels, workloads, candidates, blocks, and timestamps are paired or nested. No panel is oracle truth, and no general reliability or superiority inference is supported.\n",
            analysis.classification
        )
        .as_bytes(),
    )?;
    write_manifest(output)?;
    Ok(analysis)
}

fn write_manifest(output: &Path) -> Result<(), ResearchError> {
    let mut paths = fs::read_dir(output)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut manifest = String::new();
    for path in paths {
        if path.file_name().is_some_and(|name| name == "SHA256SUMS") {
            continue;
        }
        let name = path
            .file_name()
            .ok_or(ResearchError::Invalid("analysis output lacks file name"))?
            .to_string_lossy();
        manifest.push_str(&format!("{:x}  {name}\n", Sha256::digest(fs::read(&path)?)));
    }
    write_new(&output.join("SHA256SUMS"), manifest.trim_end().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn protocol() -> ReferencePanelProtocol {
        toml::from_str(include_str!(
            "../../../research/protocols/m8-phase2a-v1.toml"
        ))
        .unwrap()
    }

    fn adapter() -> AdapterSummaryRecord {
        AdapterSummaryRecord {
            name: "Apple M5".into(),
            backend: "Metal".into(),
            device_type: "IntegratedGpu".into(),
            max_buffer_size: 1,
            max_storage_buffer_binding_size: 1,
            max_compute_workgroups_per_dimension: 1,
            timestamp_query_supported: true,
            timestamp_period_ns: Some(1.0),
        }
    }

    fn fixture_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "constraintforge-m8-phase2a-{name}-{}-{}",
            std::process::id(),
            now().unwrap()
        ))
    }

    fn synthetic_session(
        output: &Path,
        protocol: &ReferencePanelProtocol,
        session_index: usize,
    ) -> ReferencePanelSession {
        let mut software = current_software_metadata().unwrap();
        software.git_worktree_clean = Some(true);
        let software_for_measure = software.clone();
        run_reference_panel_session_with(
            protocol,
            &format!("synthetic-session-{:03}", session_index + 1),
            session_index,
            output,
            false,
            true,
            || Ok(software.clone()),
            |panel, config| {
                assert!(output.join("session.json").is_file());
                let candidate_index = tiled_candidate_ids()
                    .iter()
                    .position(|candidate| candidate == &config.candidate_id)
                    .unwrap();
                let panel_b = panel.panel_id == "B";
                let rank_component = if panel_b {
                    9 - candidate_index
                } else {
                    candidate_index
                };
                let value = 100.0 + rank_component as f64 * 10.0 + config.block_index as f64;
                Ok((
                    adapter(),
                    software_for_measure.clone(),
                    ReferenceBlockResult {
                        block_index: config.block_index,
                        order_index: config.order_index,
                        candidate_id: config.candidate_id.clone(),
                        started_utc: 1,
                        ended_utc: 2,
                        raw_samples_ns: vec![value; config.samples as usize],
                    },
                    1,
                    1,
                ))
            },
        )
        .unwrap()
    }

    fn assert_valid_manifest(output: &Path, expected_entries: usize) {
        let bytes = fs::read(output.join("SHA256SUMS")).unwrap();
        assert!(bytes.ends_with(b"\n"));
        assert!(!bytes.ends_with(b"\n\n"));
        let text = std::str::from_utf8(&bytes).unwrap();
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), expected_entries);
        assert!(!lines.iter().any(|line| line.is_empty()));
        let mut paths = BTreeSet::new();
        let mut previous = None;
        for line in lines {
            let (hash, relative) = line.split_once("  ").unwrap();
            assert_eq!(hash.len(), 64);
            assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_ne!(relative, "SHA256SUMS");
            assert!(paths.insert(relative));
            assert!(previous.is_none_or(|saved| saved < relative));
            previous = Some(relative);
            let path = output.join(relative);
            assert!(path.is_file());
            assert_eq!(
                hash,
                format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
            );
        }
        let verified = Command::new("shasum")
            .args(["-a", "256", "-c", "SHA256SUMS"])
            .current_dir(output)
            .output()
            .unwrap();
        assert!(verified.status.success());
        assert!(!String::from_utf8_lossy(&verified.stderr).contains("WARNING"));
    }

    fn rewrite_session_source(session: &Path, change: impl FnOnce(&mut SoftwareMetadata)) {
        let state_path = session.join("session.json");
        let mut state: ReferencePanelSession =
            serde_json::from_slice(&fs::read(&state_path).unwrap()).unwrap();
        change(&mut state.software);
        for relative in &state.record_files {
            let path = session.join(relative);
            let mut record: CandidatePanelRecord =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            record.source_revision = state.software.source_revision.clone();
            record.source_tree_sha256 = state.software.source_tree_sha256.clone();
            atomic_json(&path, &record).unwrap();
        }
        atomic_json(&state_path, &state).unwrap();
    }

    fn source_tree_from_manifest(software: &SoftwareMetadata) -> String {
        let mut tree = Sha256::new();
        for entry in &software.source_manifest_sha256 {
            tree.update(entry.path.as_bytes());
            tree.update([0]);
            tree.update(entry.sha256.as_bytes());
            tree.update([0xff]);
        }
        format!("{:x}", tree.finalize())
    }

    fn directory_digest(root: &Path) -> String {
        fn collect(root: &Path, path: &Path, files: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    collect(root, &path, files);
                } else {
                    files.push(path.strip_prefix(root).unwrap().to_path_buf());
                }
            }
        }
        let mut files = Vec::new();
        collect(root, root, &mut files);
        files.sort();
        let mut digest = Sha256::new();
        for relative in files {
            digest.update(relative.to_string_lossy().as_bytes());
            digest.update([0]);
            digest.update(fs::read(root.join(relative)).unwrap());
            digest.update([0xff]);
        }
        format!("{:x}", digest.finalize())
    }

    #[test]
    fn protocol_and_all_five_plans_are_exact_and_independent() {
        let protocol = protocol();
        protocol.validate().unwrap();
        assert_eq!(
            protocol.sha256().unwrap(),
            "d8e80f9a37359dadb86aa798ba5e798f22237e4a7cf0a3ee625eba239005939e"
        );
        let plans = (0..5)
            .map(|index| reference_panel_plan(&protocol, index).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.session_seed)
                .collect::<BTreeSet<_>>()
                .len(),
            5
        );
        assert_eq!(
            plans
                .iter()
                .filter(|plan| plan.panels[0].panel_id == "A")
                .count(),
            3
        );
        assert_eq!(
            plans
                .iter()
                .filter(|plan| plan.panels[0].panel_id == "B")
                .count(),
            2
        );
        for plan in plans {
            assert_eq!(plan.panels.len(), 4);
            assert_eq!(plan.expected_candidate_records, 40);
            assert_eq!(plan.expected_blocks, 120);
            assert_eq!(plan.expected_timed_samples, 1200);
            assert_eq!(plan.expected_warmups, 360);
            assert_eq!(plan.expected_correctness_admissions, 120);
            assert_eq!(plan.expected_total_candidate_invocations, 1680);
            for workload in protocol.workloads.iter() {
                let a = plan
                    .panels
                    .iter()
                    .find(|p| p.workload == *workload && p.panel_id == "A")
                    .unwrap();
                let b = plan
                    .panels
                    .iter()
                    .find(|p| p.workload == *workload && p.panel_id == "B")
                    .unwrap();
                assert_ne!(a.stream_seed_sha256, b.stream_seed_sha256);
                assert_ne!(a.blocks[0].candidate_ids, b.blocks[0].candidate_ids);
            }
        }
    }

    #[test]
    fn production_runner_mock_is_complete_and_analysis_reproduces() {
        let root = fixture_root("e2e");
        let protocol = protocol();
        let session = root.join("session-001");
        let state = synthetic_session(&session, &protocol, 0);
        assert!(state.synthetic);
        assert_eq!(state.record_files.len(), 40);
        assert_eq!(state.attempts.len(), 120);
        let validation = validate_reference_panel_results(&session).unwrap();
        assert!(validation.valid);
        assert_eq!(
            (
                validation.candidate_records,
                validation.blocks,
                validation.timed_samples
            ),
            (40, 120, 1200)
        );
        let a = root.join("analysis-a");
        let b = root.join("analysis-b");
        let result = analyze_reference_panels(&session, &a, true).unwrap();
        analyze_reference_panels(&session, &b, true).unwrap();
        assert_eq!(
            result.classification,
            "SINGLE-SESSION QUALIFICATION / DESCRIPTIVE ONLY"
        );
        let rank_agreement = fs::read_to_string(a.join("rank-agreement.csv")).unwrap();
        assert_eq!(rank_agreement.matches(",-1,-1,").count(), 2);
        let best_agreement = fs::read_to_string(a.join("best-set-agreement.csv")).unwrap();
        assert_eq!(best_agreement.matches(",false,0").count(), 2);
        assert_valid_manifest(&a, 10);
        assert_valid_manifest(&b, 10);
        for name in [
            "panel-candidates.csv",
            "panel-rankings.csv",
            "best-set-agreement.csv",
            "rank-agreement.csv",
            "cross-panel-regret.csv",
            "reference-variability.csv",
            "validation.json",
            "session-summary.json",
            "resolved-protocol.json",
            "summary.md",
            "SHA256SUMS",
        ] {
            assert_eq!(
                fs::read(a.join(name)).unwrap(),
                fs::read(b.join(name)).unwrap()
            );
        }
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(a.join("SHA256SUMS")).unwrap())
            ),
            "aff5f8deb82d19c664003e4ae0eb283a3091b0d5825ab9c873cdb3a58dc58965"
        );
        assert!(analyze_reference_panels(&session, &root.join("full"), false).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn full_study_requires_five_sessions_and_reproduces() {
        let root = fixture_root("full");
        let protocol = protocol();
        for index in 0..5 {
            synthetic_session(
                &root.join(format!("session-{:03}", index + 1)),
                &protocol,
                index,
            );
        }
        let a = root.join("analysis-a");
        let b = root.join("analysis-b");
        let result = analyze_reference_panels(&root, &a, false).unwrap();
        analyze_reference_panels(&root, &b, false).unwrap();
        assert_eq!(
            (
                result.sessions,
                result.workload_pairs,
                result.candidate_rows
            ),
            (5, 10, 200)
        );
        assert_eq!(
            fs::read(a.join("SHA256SUMS")).unwrap(),
            fs::read(b.join("SHA256SUMS")).unwrap()
        );
        assert_valid_manifest(&a, 10);
        assert_valid_manifest(&b, 10);
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(a.join("SHA256SUMS")).unwrap())
            ),
            "24457a8ab5b223782c88f436b82168c1777cde41e9cede246843505df0c040aa"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn campaign_validation_rejects_every_mixed_or_incomplete_source_identity() {
        type SourceMutation = Box<dyn Fn(&mut SoftwareMetadata)>;
        let protocol = protocol();
        let cases: Vec<(&str, SourceMutation)> = vec![
            (
                "source_revision",
                Box::new(|software| software.source_revision = "different-revision".into()),
            ),
            (
                "source_tree_sha256",
                Box::new(|software| {
                    software.source_manifest_sha256[0].sha256 = "0".repeat(64);
                    software.source_tree_sha256 = Some(source_tree_from_manifest(software));
                }),
            ),
            (
                "missing measurement source revision",
                Box::new(|software| software.source_revision.clear()),
            ),
            (
                "missing measurement source tree SHA-256",
                Box::new(|software| software.source_tree_sha256 = None),
            ),
            (
                "inconsistent with its per-file manifest",
                Box::new(|software| software.source_manifest_sha256[0].sha256 = "0".repeat(64)),
            ),
            (
                "implementation version",
                Box::new(|software| {
                    software.research_implementation_version = Some("different-version".into())
                }),
            ),
            (
                "cargo_lock_sha256",
                Box::new(|software| software.cargo_lock_sha256 = Some("0".repeat(64))),
            ),
            (
                "rust_toolchain_sha256",
                Box::new(|software| software.rust_toolchain_sha256 = Some("0".repeat(64))),
            ),
            (
                "missing measurement source per-file manifest",
                Box::new(|software| software.source_manifest_sha256.clear()),
            ),
        ];
        for (case, change) in cases {
            let root = fixture_root(case.replace(' ', "-").as_str());
            for index in 0..5 {
                synthetic_session(
                    &root.join(format!("session-{:03}", index + 1)),
                    &protocol,
                    index,
                );
            }
            rewrite_session_source(&root.join("session-005"), change);
            let before = directory_digest(&root);
            let validation = validate_reference_panel_results(&root).unwrap();
            assert!(!validation.valid, "case unexpectedly passed: {case}");
            assert!(
                validation.errors.iter().any(|error| {
                    error.contains("synthetic-session-005") && error.contains(case)
                }),
                "case lacked an explicit session/field error: {case}: {:?}",
                validation.errors
            );
            assert_eq!(directory_digest(&root), before);
            let output = root.join("analysis");
            assert!(analyze_reference_panels(&root, &output, false).is_err());
            assert!(!output.exists());
            assert_eq!(directory_digest(&root), before);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn validation_rejects_missing_duplicate_and_invalid_measurements() {
        let root = fixture_root("invalid");
        let protocol = protocol();
        let session = root.join("session");
        let state = synthetic_session(&session, &protocol, 0);
        let path = session.join(&state.record_files[0]);
        let original = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        fs::write(&path, &original).unwrap();
        let mut record: CandidatePanelRecord = serde_json::from_slice(&original).unwrap();
        record.blocks[0].raw_samples_ns[0] = 0.0;
        atomic_json(&path, &record).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        record.blocks[0].raw_samples_ns[0] = 1.0;
        record.blocks.push(record.blocks[0].clone());
        atomic_json(&path, &record).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        fs::write(&path, &original).unwrap();
        let mut record: CandidatePanelRecord = serde_json::from_slice(&original).unwrap();
        record.adapter.as_mut().unwrap().timestamp_period_ns = Some(0.0);
        atomic_json(&path, &record).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        record.adapter.as_mut().unwrap().timestamp_period_ns = Some(1.0);
        record.source_revision = "different-source".into();
        atomic_json(&path, &record).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        fs::write(&path, &original).unwrap();
        let protocol_path = session.join("resolved-protocol.toml");
        let mut changed = protocol.clone();
        changed.version = "different-protocol".into();
        fs::write(&protocol_path, toml::to_string_pretty(&changed).unwrap()).unwrap();
        assert!(!validate_reference_panel_results(&session).unwrap().valid);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ambiguous_attempt_refuses_remeasurement_and_committed_attempt_reconciles() {
        let root = fixture_root("attempt");
        let protocol = protocol();
        let mut state = synthetic_session(&root, &protocol, 0);
        state.attempts[0].status = "started_or_possibly_executed".into();
        state.attempts[0].ended_utc = None;
        reconcile_attempts(&root, &mut state).unwrap();
        assert_eq!(state.attempts[0].status, "completed_and_durably_recorded");
        let panel = state
            .plan
            .panels
            .iter()
            .find(|panel| {
                panel.panel_id == state.attempts[1].panel_id
                    && panel.workload == state.attempts[1].workload
            })
            .unwrap();
        let path = root.join(record_path(panel, &state.attempts[1].candidate_id));
        let mut record: CandidatePanelRecord =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record
            .blocks
            .retain(|block| block.block_index != state.attempts[1].block_index);
        atomic_json(&path, &record).unwrap();
        state.attempts[1].status = "started_or_possibly_executed".into();
        state.attempts[1].ended_utc = None;
        assert!(reconcile_attempts(&root, &mut state).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn storage_failure_precedes_all_mock_invocations_and_overwrite_refuses() {
        let root = fixture_root("storage");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("panels"), b"not a directory").unwrap();
        let protocol = protocol();
        let calls = std::cell::Cell::new(0);
        let mut software = current_software_metadata().unwrap();
        software.git_worktree_clean = Some(true);
        let result = run_reference_panel_session_with(
            &protocol,
            "synthetic",
            0,
            &root,
            true,
            true,
            || Ok(software.clone()),
            |_, _| {
                calls.set(calls.get() + 1);
                Err("must not execute".into())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
        fs::remove_dir_all(root).unwrap();

        let root = fixture_root("overwrite");
        synthetic_session(&root, &protocol, 0);
        let calls = std::cell::Cell::new(0);
        let result = run_reference_panel_session_with(
            &protocol,
            "synthetic",
            0,
            &root,
            false,
            true,
            || Ok(software.clone()),
            |_, _| {
                calls.set(calls.get() + 1);
                Err("must not execute".into())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn measurement_failure_leaves_one_ambiguous_durable_intent() {
        let root = fixture_root("measurement-failure");
        let protocol = protocol();
        let calls = std::cell::Cell::new(0);
        let mut software = current_software_metadata().unwrap();
        software.git_worktree_clean = Some(true);
        let result = run_reference_panel_session_with(
            &protocol,
            "synthetic",
            0,
            &root,
            false,
            true,
            || Ok(software.clone()),
            |_, _| {
                calls.set(calls.get() + 1);
                Err("synthetic correctness mismatch".into())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 1);
        let state: ReferencePanelSession =
            serde_json::from_slice(&fs::read(root.join("session.json")).unwrap()).unwrap();
        assert_eq!(state.attempts.len(), 1);
        assert_eq!(state.attempts[0].status, "started_or_possibly_executed");
        assert!(
            run_reference_panel_session_with(
                &protocol,
                "synthetic",
                0,
                &root,
                true,
                true,
                || Ok(software.clone()),
                |_, _| panic!("ambiguous attempt must not be remeasured"),
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ties_undefined_correlations_and_cross_panel_regret_are_exact() {
        let values = BTreeMap::from([("a".into(), 1.0), ("b".into(), 1.0), ("c".into(), 2.0)]);
        let ranks = ranks(&values);
        assert_eq!(ranks["a"], 1.5);
        assert_eq!(ranks["b"], 1.5);
        assert!(correlation(&[1.0, 1.0], &[2.0, 2.0]).is_none());
        assert!(kendall_tau_b(&[1.0, 1.0], &[2.0, 2.0]).is_none());
        assert_eq!((12.0_f64 - 10.0) / 10.0, 0.2);
    }
}

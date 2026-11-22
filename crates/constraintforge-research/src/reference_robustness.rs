use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

use constraintforge_bench::current_software_metadata;
use constraintforge_tuner::{TuningResult, tiled_candidate_ids};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ResearchError, find_named,
    prospective::{
        CandidateReferenceRecord, ProspectiveSessionRecord, validate_prospective_results,
    },
    write_new,
};

const PLAN_SCHEMA: &str = "constraintforge-m8-reference-reliability-plan-v1";
const OUTPUT_SCHEMA: &str = "constraintforge-m8-reference-reliability-analysis-v1";
const SUBSETS: [&str; 6] = ["P10", "P20", "P30", "LOBO-0", "LOBO-1", "LOBO-2"];
type PairedKey = (String, [usize; 3], usize);
type BestKey = (String, [usize; 3]);
type SessionSummaryRow = (usize, [usize; 3], usize, usize, f64, f64);

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceReliabilityPlan {
    schema: String,
    id: String,
    version: String,
    phase: String,
    frozen: bool,
    classification: String,
    research_question: String,
    dataset_collection_status: String,
    accepted_sessions: Vec<String>,
    excluded_session: String,
    accepted_session_schema: String,
    accepted_reference_schema: String,
    required_m7_source_revision: String,
    required_m7_execution_revision: String,
    required_m7_source_tree_sha256: String,
    required_m7_protocol_sha256: String,
    required_m7_resolved_protocol_sha256: String,
    required_adapter_name: String,
    required_backend: String,
    workloads: Vec<[usize; 3]>,
    candidates_per_panel: usize,
    reference_blocks_per_candidate: usize,
    samples_per_block: usize,
    prefixes: Vec<usize>,
    block_omissions: Vec<usize>,
    subset_ordering: String,
    reference_estimator: String,
    tie_rule: String,
    rank_metrics: Vec<String>,
    primary_diagnostics: Vec<String>,
    secondary_diagnostics: Vec<String>,
    statistical_unit: String,
    output_schema: String,
    deterministic_ordering: String,
    missing_data_rule: String,
    full_panel_crosscheck: String,
    input_manifest: String,
    frozen_m7_paired_output: String,
    frozen_m7_reference_output: String,
}

impl ReferenceReliabilityPlan {
    pub fn parse(path: &Path) -> Result<(Self, String), ResearchError> {
        let bytes = fs::read(path)?;
        let plan: Self = toml::from_str(
            std::str::from_utf8(&bytes)
                .map_err(|_| ResearchError::Invalid("M8 plan is not UTF-8"))?,
        )?;
        plan.validate()?;
        Ok((plan, format!("{:x}", Sha256::digest(bytes))))
    }

    fn validate(&self) -> Result<(), ResearchError> {
        let candidates = tiled_candidate_ids();
        let expected_sessions = [
            "m7-phase2a-session-001-requal-01",
            "m7-phase2a-session-002",
            "m7-phase2a-session-003",
            "m7-phase2a-session-004",
            "m7-phase2a-session-005",
        ];
        if self.schema != PLAN_SCHEMA
            || self.id != "m8-reference-reliability-retrospective-v1"
            || self.version != "1.0.0"
            || self.phase != "frozen-analysis"
            || !self.frozen
            || self.classification != "RETROSPECTIVE / EXPLORATORY"
            || self
                .accepted_sessions
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != expected_sessions
            || self.excluded_session != "m7-phase2a-session-001"
            || self.accepted_session_schema != "constraintforge-m7-prospective-session-v2"
            || self.accepted_reference_schema != "constraintforge-m7-candidate-reference-v2"
            || self.required_m7_source_revision != "57f364bcc66f12d1dcab769ae7b05fbad83abbf8"
            || self.required_m7_execution_revision != "948610fe1182d9be5ea319fdaa07c17a9915029e"
            || self.required_m7_source_tree_sha256
                != "a788bb0262ee91c675182c0e5f10a53e693c5e8bfbd615ca16403c72fb2f048d"
            || self.required_m7_protocol_sha256
                != "19aae7b90745c01405134201f100a58700b7b698b9661391bcec00724d09923c"
            || self.required_m7_resolved_protocol_sha256
                != "581514d3e924958648d20dcda52bc746add4c24f68d5296ae5e7675d08e648e5"
            || self.required_adapter_name != "Apple M5"
            || self.required_backend != "Metal"
            || self.workloads != [[7, 19, 11], [256, 64, 384]]
            || self.candidates_per_panel != 10
            || candidates.len() != 10
            || self.reference_blocks_per_candidate != 3
            || self.samples_per_block != 10
            || self.prefixes != [10, 20, 30]
            || self.block_omissions != [0, 1, 2]
            || self.output_schema != OUTPUT_SCHEMA
        {
            return Err(ResearchError::Invalid(
                "M8 plan differs from the authorized v1 design",
            ));
        }
        for value in [
            &self.input_manifest,
            &self.frozen_m7_paired_output,
            &self.frozen_m7_reference_output,
        ] {
            if !safe_relative(Path::new(value)) {
                return Err(ResearchError::Invalid(
                    "M8 plan contains an unsafe evidence path",
                ));
            }
        }
        if self.research_question.is_empty()
            || self.dataset_collection_status.is_empty()
            || self.subset_ordering.is_empty()
            || self.reference_estimator.is_empty()
            || self.tie_rule.is_empty()
            || self.rank_metrics.len() != 4
            || self.primary_diagnostics.is_empty()
            || self.secondary_diagnostics.is_empty()
            || self.statistical_unit.is_empty()
            || self.deterministic_ordering.is_empty()
            || self.missing_data_rule.is_empty()
            || self.full_panel_crosscheck.is_empty()
        {
            return Err(ResearchError::Invalid(
                "M8 plan has incomplete scientific definitions",
            ));
        }
        Ok(())
    }
}

fn safe_relative(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

#[derive(Clone, Debug)]
struct CandidatePanel {
    id: String,
    blocks: [Vec<f64>; 3],
}

#[derive(Clone, Debug)]
struct SubsetPanel {
    name: &'static str,
    values: Vec<(String, f64)>,
    ranks: Vec<f64>,
    best: Vec<String>,
    gap: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReferenceReliabilityAnalysis {
    pub schema: String,
    pub classification: String,
    pub sessions: usize,
    pub panels: usize,
    pub candidates: usize,
    pub reference_blocks: usize,
    pub reference_samples: usize,
    pub search_trials: usize,
    pub paired_effects: usize,
    pub full_panel_crosscheck: String,
    pub m7_qualified_source_revision: String,
    pub m7_execution_revision: String,
    pub m7_source_tree_sha256: String,
    pub m7_protocol_file_sha256: String,
    pub m7_resolved_protocol_sha256: String,
    pub m8_plan_sha256: String,
    pub m8_analysis_revision: String,
    pub m8_analysis_source_tree_sha256: Option<String>,
    pub input_manifest_sha256: String,
    pub input_files: usize,
    pub output_files_excluding_manifest: usize,
}

pub fn reference_robustness(
    plan_path: &Path,
    input: &Path,
    output: &Path,
) -> Result<ReferenceReliabilityAnalysis, ResearchError> {
    if output.exists() {
        return Err(ResearchError::OutputExists(output.into()));
    }
    let (plan, plan_sha) = ReferenceReliabilityPlan::parse(plan_path)?;
    let validation = validate_prospective_results(input)?;
    if !validation.valid
        || validation.sessions != 5
        || validation.search_trials != 40
        || validation.reference_records != 100
    {
        return Err(ResearchError::Invalid(
            "M8 requires the complete validated five-session M7 input",
        ));
    }
    let (input_manifest, input_files, input_manifest_sha) =
        verify_input_manifest(input, Path::new(&plan.input_manifest))?;
    let mut states = Vec::new();
    for path in find_named(input, "session.json")? {
        let state: ProspectiveSessionRecord = serde_json::from_slice(&fs::read(&path)?)?;
        let root = path
            .parent()
            .ok_or(ResearchError::Invalid("invalid session path"))?
            .to_path_buf();
        states.push((root, state));
    }
    states.sort_by_key(|(_, state)| state.session_index);
    validate_states(&plan, &states)?;

    let frozen_pairs = load_frozen_pairs(Path::new(&plan.frozen_m7_paired_output))?;
    let frozen_bests = load_frozen_bests(Path::new(&plan.frozen_m7_reference_output))?;
    let mut candidate_csv = String::from(
        "schema,session,session_index,m,k,n,subset,candidate,reference_median_ns,rank,is_best,within_block_cv_max,between_block_median_cv,stored_order\n",
    );
    let mut best_csv = String::from(
        "schema,session,session_index,m,k,n,subset,best_set,full_best_retained,distinct_best_candidates,best_runner_up_gap_ns\n",
    );
    let mut rank_csv = String::from(
        "schema,session,session_index,m,k,n,subset,reference_subset,spearman_rho,kendall_tau_b,top1_set_agreement,top3_jaccard,pairs,dependent_comparison\n",
    );
    let mut regret_csv = String::from(
        "schema,session,session_index,m,k,n,budget,subset,resource_aware_candidate,random_candidate,resource_aware_regret,random_regret,paired_difference,change_from_p30,sign_vs_p30,shared_denominator,statistical_unit\n",
    );
    let mut summary_rows = Vec::<SessionSummaryRow>::new();
    let mut paired_count = 0;

    for (root, state) in &states {
        for workload in &plan.workloads {
            let candidates = load_panel(&plan, root, state, *workload)?;
            let subsets = subset_panels(&candidates)?;
            let full = &subsets[2];
            let full_best = full.best.iter().cloned().collect::<BTreeSet<_>>();
            let mut union_best = BTreeSet::new();
            for subset in &subsets {
                union_best.extend(subset.best.iter().cloned());
            }
            let selections = load_selections(root, state, *workload)?;
            for subset in &subsets {
                for ((id, value), rank) in subset.values.iter().zip(&subset.ranks) {
                    let candidate = candidates
                        .iter()
                        .find(|candidate| candidate.id == *id)
                        .ok_or(ResearchError::Invalid("candidate disappeared"))?;
                    candidate_csv.push_str(&format!(
                        "{OUTPUT_SCHEMA},{},{},{},{},{},{},{},{},{},{},{},{},true\n",
                        state.session_id,
                        state.session_index,
                        workload[0],
                        workload[1],
                        workload[2],
                        subset.name,
                        id,
                        value,
                        rank,
                        subset.best.contains(id),
                        within_block_cv_max(candidate)?,
                        between_block_cv(candidate)?
                    ));
                }
                let retained = subset
                    .best
                    .iter()
                    .any(|candidate| full_best.contains(candidate));
                best_csv.push_str(&format!(
                    "{OUTPUT_SCHEMA},{},{},{},{},{},{},{},{},{},{}\n",
                    state.session_id,
                    state.session_index,
                    workload[0],
                    workload[1],
                    workload[2],
                    subset.name,
                    subset.best.join("|"),
                    retained,
                    union_best.len(),
                    option_number(subset.gap)
                ));
                if subset.name != "P30" {
                    let (spearman, kendall) = rank_agreement(&subset.values, &full.values);
                    rank_csv.push_str(&format!(
                        "{OUTPUT_SCHEMA},{},{},{},{},{},{},P30,{},{},{},{},{},true\n",
                        state.session_id,
                        state.session_index,
                        workload[0],
                        workload[1],
                        workload[2],
                        subset.name,
                        option_number(spearman),
                        option_number(kendall),
                        subset.best == full.best,
                        top_k_jaccard(&subset.values, &full.values, 3),
                        pair_count(candidates.len())
                    ));
                }
            }
            for block_index in 0..3 {
                let block = block_panel(&candidates, block_index)?;
                let (spearman, kendall) = rank_agreement(&block.values, &full.values);
                rank_csv.push_str(&format!(
                    "{OUTPUT_SCHEMA},{},{},{},{},{},{},P30,{},{},{},{},{},true\n",
                    state.session_id,
                    state.session_index,
                    workload[0],
                    workload[1],
                    workload[2],
                    block.name,
                    option_number(spearman),
                    option_number(kendall),
                    block.best == full.best,
                    top_k_jaccard(&block.values, &full.values, 3),
                    pair_count(candidates.len())
                ));
            }
            crosscheck_best(state, *workload, &full.best, &frozen_bests)?;
            for budget in [2usize, 8] {
                let aware = selections
                    .get(&(budget, "resource-aware".to_string()))
                    .ok_or(ResearchError::Invalid("missing resource-aware selection"))?;
                let random = selections
                    .get(&(budget, "random".to_string()))
                    .ok_or(ResearchError::Invalid("missing random selection"))?;
                let p30 = paired_effect(full, aware, random)?;
                let key = (state.session_id.clone(), *workload, budget);
                let frozen = frozen_pairs
                    .get(&key)
                    .ok_or(ResearchError::Invalid("missing frozen M7 paired effect"))?;
                if p30.to_bits() != frozen.to_bits() {
                    return Err(ResearchError::Invalid(
                        "M8 P30 paired effect does not reproduce frozen M7",
                    ));
                }
                let mut worst: f64 = 0.0;
                let mut sign_changes = 0;
                for subset in &subsets {
                    let (aware_regret, random_regret, paired) = regrets(subset, aware, random)?;
                    let change = paired - p30;
                    worst = worst.max(change.abs());
                    let sign = sign_label(paired, p30);
                    if sign == "changed" {
                        sign_changes += 1;
                    }
                    regret_csv.push_str(&format!(
                        "{OUTPUT_SCHEMA},{},{},{},{},{},{},{},{},{},{},{},{},{},{},true,session\n",
                        state.session_id,
                        state.session_index,
                        workload[0],
                        workload[1],
                        workload[2],
                        budget,
                        subset.name,
                        aware,
                        random,
                        aware_regret,
                        random_regret,
                        paired,
                        change,
                        sign
                    ));
                    paired_count += 1;
                }
                summary_rows.push((
                    state.session_index,
                    *workload,
                    budget,
                    sign_changes,
                    worst,
                    p30,
                ));
            }
        }
    }

    fs::create_dir(output)?;
    write_new(
        &output.join("input-files.sha256"),
        input_manifest.as_bytes(),
    )?;
    write_new(
        &output.join("validation.json"),
        &serde_json::to_vec_pretty(&validation)?,
    )?;
    write_new(
        &output.join("candidate-subsets.csv"),
        candidate_csv.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("best-membership.csv"),
        best_csv.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("rank-stability.csv"),
        rank_csv.trim_end().as_bytes(),
    )?;
    write_new(
        &output.join("regret-sensitivity.csv"),
        regret_csv.trim_end().as_bytes(),
    )?;
    let session_summary = session_summary_csv(&summary_rows);
    write_new(
        &output.join("session-summary.csv"),
        session_summary.trim_end().as_bytes(),
    )?;
    let software = current_software_metadata()?;
    let analysis = ReferenceReliabilityAnalysis {
        schema: OUTPUT_SCHEMA.into(),
        classification: plan.classification.clone(),
        sessions: 5,
        panels: 10,
        candidates: 100,
        reference_blocks: 300,
        reference_samples: 3_000,
        search_trials: 40,
        paired_effects: paired_count,
        full_panel_crosscheck: "PASS: 10 empirical-best sets and 20 paired effects".into(),
        m7_qualified_source_revision: plan.required_m7_source_revision.clone(),
        m7_execution_revision: plan.required_m7_execution_revision.clone(),
        m7_source_tree_sha256: plan.required_m7_source_tree_sha256.clone(),
        m7_protocol_file_sha256: plan.required_m7_protocol_sha256.clone(),
        m7_resolved_protocol_sha256: plan.required_m7_resolved_protocol_sha256.clone(),
        m8_plan_sha256: plan_sha,
        m8_analysis_revision: software.source_revision,
        m8_analysis_source_tree_sha256: software.source_tree_sha256,
        input_manifest_sha256: input_manifest_sha,
        input_files,
        output_files_excluding_manifest: 9,
    };
    write_new(
        &output.join("analysis.json"),
        &serde_json::to_vec_pretty(&analysis)?,
    )?;
    write_new(
        &output.join("report.md"),
        report(&analysis, &summary_rows).as_bytes(),
    )?;
    write_manifest(output)?;
    Ok(analysis)
}

fn validate_states(
    plan: &ReferenceReliabilityPlan,
    states: &[(PathBuf, ProspectiveSessionRecord)],
) -> Result<(), ResearchError> {
    if states.len() != 5 {
        return Err(ResearchError::Invalid(
            "M8 input must contain five sessions",
        ));
    }
    let ids = states
        .iter()
        .map(|(_, state)| state.session_id.as_str())
        .collect::<Vec<_>>();
    if ids
        != plan
            .accepted_sessions
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
        || ids.contains(&plan.excluded_session.as_str())
    {
        return Err(ResearchError::Invalid(
            "M8 accepted/excluded session identity mismatch",
        ));
    }
    for (index, (_, state)) in states.iter().enumerate() {
        if state.session_index != index
            || state.schema != plan.accepted_session_schema
            || state.phase != "complete"
            || !state.selections_sealed
            || state.selection_seal_sha256.is_none()
            || state.search.len() != 8
            || state.reference_files.len() != 20
            || state.reference_attempts.len() != 60
            || state
                .reference_attempts
                .iter()
                .any(|attempt| attempt.status != "completed_and_durably_recorded")
            || state.software.source_revision != plan.required_m7_execution_revision
            || state.software.source_tree_sha256.as_deref()
                != Some(&plan.required_m7_source_tree_sha256)
            || state.protocol_sha256 != plan.required_m7_resolved_protocol_sha256
        {
            return Err(ResearchError::Invalid(
                "M8 session provenance or completion mismatch",
            ));
        }
    }
    Ok(())
}

fn load_panel(
    plan: &ReferenceReliabilityPlan,
    root: &Path,
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
) -> Result<Vec<CandidatePanel>, ResearchError> {
    let legal = tiled_candidate_ids().into_iter().collect::<BTreeSet<_>>();
    let mut found = BTreeMap::new();
    for relative in &state.reference_files {
        let record: CandidateReferenceRecord =
            serde_json::from_slice(&fs::read(root.join(relative))?)?;
        if record.workload != workload {
            continue;
        }
        if record.schema != plan.accepted_reference_schema
            || record.session_id != state.session_id
            || record.session_index != state.session_index
            || record.protocol_sha256 != plan.required_m7_resolved_protocol_sha256
            || record.source_revision != plan.required_m7_execution_revision
            || record.source_tree_sha256.as_deref() != Some(&plan.required_m7_source_tree_sha256)
            || record.status != "complete"
            || record.error.is_some()
            || !legal.contains(&record.candidate_id)
            || record.adapter.as_ref().is_none_or(|adapter| {
                adapter.name != plan.required_adapter_name
                    || adapter.backend != plan.required_backend
            })
            || record.blocks.len() != 3
        {
            return Err(ResearchError::Invalid(
                "M8 reference identity or completeness mismatch",
            ));
        }
        let mut blocks: [Option<Vec<f64>>; 3] = [None, None, None];
        for block in record.blocks {
            if block.block_index >= 3
                || block.candidate_id != record.candidate_id
                || block.raw_samples_ns.len() != 10
                || block
                    .raw_samples_ns
                    .iter()
                    .any(|sample| !sample.is_finite() || *sample <= 0.0)
                || blocks[block.block_index]
                    .replace(block.raw_samples_ns)
                    .is_some()
            {
                return Err(ResearchError::Invalid(
                    "M8 reference block is invalid or duplicated",
                ));
            }
        }
        let panel = CandidatePanel {
            id: record.candidate_id.clone(),
            blocks: [
                blocks[0]
                    .take()
                    .ok_or(ResearchError::Invalid("missing reference block"))?,
                blocks[1]
                    .take()
                    .ok_or(ResearchError::Invalid("missing reference block"))?,
                blocks[2]
                    .take()
                    .ok_or(ResearchError::Invalid("missing reference block"))?,
            ],
        };
        if found.insert(record.candidate_id, panel).is_some() {
            return Err(ResearchError::Invalid("duplicate reference candidate"));
        }
    }
    if found.len() != 10 || found.keys().cloned().collect::<BTreeSet<_>>() != legal {
        return Err(ResearchError::Invalid("unexpected candidate coverage"));
    }
    Ok(found.into_values().collect())
}

fn subset_panels(candidates: &[CandidatePanel]) -> Result<Vec<SubsetPanel>, ResearchError> {
    let mut panels = Vec::new();
    for (position, name) in SUBSETS.iter().enumerate() {
        let mut values = Vec::new();
        for candidate in candidates {
            let samples = if position < 3 {
                candidate
                    .blocks
                    .iter()
                    .flatten()
                    .copied()
                    .take([10, 20, 30][position])
                    .collect::<Vec<_>>()
            } else {
                let omitted = position - 3;
                candidate
                    .blocks
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != omitted)
                    .flat_map(|(_, block)| block.iter().copied())
                    .collect()
            };
            values.push((candidate.id.clone(), median(&samples)?));
        }
        values.sort_by(|left, right| left.0.cmp(&right.0));
        let ranks = average_ranks(&values.iter().map(|(_, value)| *value).collect::<Vec<_>>());
        let minimum = values
            .iter()
            .map(|(_, value)| *value)
            .min_by(f64::total_cmp)
            .ok_or(ResearchError::Invalid("empty panel"))?;
        let best = values
            .iter()
            .filter(|(_, value)| value.total_cmp(&minimum).is_eq())
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let gap = values
            .iter()
            .map(|(_, value)| *value)
            .filter(|value| value.total_cmp(&minimum).is_gt())
            .min_by(f64::total_cmp)
            .map(|runner| runner - minimum);
        panels.push(SubsetPanel {
            name,
            values,
            ranks,
            best,
            gap,
        });
    }
    Ok(panels)
}

fn block_panel(
    candidates: &[CandidatePanel],
    block_index: usize,
) -> Result<SubsetPanel, ResearchError> {
    if block_index >= 3 {
        return Err(ResearchError::Invalid(
            "reference block index is out of range",
        ));
    }
    let mut values = candidates
        .iter()
        .map(|candidate| {
            Ok((
                candidate.id.clone(),
                median(&candidate.blocks[block_index])?,
            ))
        })
        .collect::<Result<Vec<_>, ResearchError>>()?;
    values.sort_by(|left, right| left.0.cmp(&right.0));
    let ranks = average_ranks(&values.iter().map(|(_, value)| *value).collect::<Vec<_>>());
    let minimum = values
        .iter()
        .map(|(_, value)| *value)
        .min_by(f64::total_cmp)
        .ok_or(ResearchError::Invalid("empty block panel"))?;
    let best = values
        .iter()
        .filter(|(_, value)| value.total_cmp(&minimum).is_eq())
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    Ok(SubsetPanel {
        name: ["BLOCK-0", "BLOCK-1", "BLOCK-2"][block_index],
        values,
        ranks,
        best,
        gap: None,
    })
}

fn load_selections(
    root: &Path,
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
) -> Result<BTreeMap<(usize, String), String>, ResearchError> {
    let mut selections = BTreeMap::new();
    for search in state
        .search
        .iter()
        .filter(|search| search.plan.workload == workload)
    {
        let relative = search
            .result_file
            .as_ref()
            .ok_or(ResearchError::Invalid("missing sealed search result"))?;
        let result: TuningResult = serde_json::from_slice(&fs::read(root.join(relative))?)?;
        if !result.complete || !result.errors.is_empty() || result.trial_id != search.plan.trial_id
        {
            return Err(ResearchError::Invalid("invalid sealed search result"));
        }
        if selections
            .insert(
                (search.plan.budget, search.plan.policy_id.clone()),
                result.search.selected_candidate_id,
            )
            .is_some()
        {
            return Err(ResearchError::Invalid(
                "duplicate workload/budget/policy selection",
            ));
        }
    }
    if selections.len() != 4 {
        return Err(ResearchError::Invalid("incomplete sealed selections"));
    }
    Ok(selections)
}

fn regrets(
    panel: &SubsetPanel,
    aware: &str,
    random: &str,
) -> Result<(f64, f64, f64), ResearchError> {
    let minimum = panel
        .values
        .iter()
        .map(|(_, value)| *value)
        .min_by(f64::total_cmp)
        .ok_or(ResearchError::Invalid("empty regret panel"))?;
    let selected = |id: &str| {
        panel
            .values
            .iter()
            .find(|(candidate, _)| candidate == id)
            .map(|(_, value)| *value)
            .ok_or(ResearchError::Invalid(
                "selected candidate absent from panel",
            ))
    };
    let aware_value = selected(aware)?;
    let random_value = selected(random)?;
    let aware_regret = (aware_value - minimum) / minimum;
    let random_regret = (random_value - minimum) / minimum;
    let paired = aware_regret - random_regret;
    let algebraic = (aware_value - random_value) / minimum;
    if (paired - algebraic).abs() > 1e-12 {
        return Err(ResearchError::Invalid(
            "shared-denominator paired regret invariant failed",
        ));
    }
    Ok((aware_regret, random_regret, paired))
}

fn paired_effect(panel: &SubsetPanel, aware: &str, random: &str) -> Result<f64, ResearchError> {
    Ok(regrets(panel, aware, random)?.2)
}

fn median(values: &[f64]) -> Result<f64, ResearchError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ResearchError::Invalid(
            "median requires positive finite samples",
        ));
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    Ok(if ordered.len() % 2 == 0 {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[middle]
    })
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

fn rank_agreement(left: &[(String, f64)], right: &[(String, f64)]) -> (Option<f64>, Option<f64>) {
    if left.len() != right.len() || left.iter().zip(right).any(|(a, b)| a.0 != b.0) {
        return (None, None);
    }
    let left_values = left.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let right_values = right.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let a = average_ranks(&left_values);
    let b = average_ranks(&right_values);
    let am = a.iter().sum::<f64>() / a.len() as f64;
    let bm = b.iter().sum::<f64>() / b.len() as f64;
    let covariance = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - am) * (y - bm))
        .sum::<f64>();
    let scale = (a.iter().map(|x| (x - am).powi(2)).sum::<f64>()
        * b.iter().map(|y| (y - bm).powi(2)).sum::<f64>())
    .sqrt();
    let spearman = (scale > 0.0).then_some(covariance / scale);
    let (mut concordant, mut discordant, mut left_ties, mut right_ties) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..left_values.len() {
        for j in i + 1..left_values.len() {
            let x = left_values[i].total_cmp(&left_values[j]);
            let y = right_values[i].total_cmp(&right_values[j]);
            match (x.is_eq(), y.is_eq()) {
                (true, true) => {}
                (true, false) => left_ties += 1.0,
                (false, true) => right_ties += 1.0,
                (false, false) if x == y => concordant += 1.0,
                _ => discordant += 1.0,
            }
        }
    }
    let denominator =
        ((concordant + discordant + left_ties) * (concordant + discordant + right_ties)).sqrt();
    (
        spearman,
        (denominator > 0.0).then_some((concordant - discordant) / denominator),
    )
}

fn top_k_jaccard(left: &[(String, f64)], right: &[(String, f64)], k: usize) -> f64 {
    let top = |values: &[(String, f64)]| {
        let mut values = values.to_vec();
        values.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        values
            .into_iter()
            .take(k)
            .map(|x| x.0)
            .collect::<BTreeSet<_>>()
    };
    let a = top(left);
    let b = top(right);
    a.intersection(&b).count() as f64 / a.union(&b).count() as f64
}

fn pair_count(n: usize) -> usize {
    n * (n - 1) / 2
}
fn option_number(value: Option<f64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "undefined".into())
}
fn sign(value: f64) -> i8 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}
fn sign_label(value: f64, baseline: f64) -> &'static str {
    if sign(value) == sign(baseline) {
        "same"
    } else {
        "changed"
    }
}

fn coefficient_of_variation(values: &[f64]) -> Result<f64, ResearchError> {
    if values.is_empty() {
        return Err(ResearchError::Invalid("CV requires observations"));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean <= 0.0 || !mean.is_finite() {
        return Err(ResearchError::Invalid("CV requires positive finite mean"));
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Ok(variance.sqrt() / mean)
}
fn within_block_cv_max(candidate: &CandidatePanel) -> Result<f64, ResearchError> {
    candidate
        .blocks
        .iter()
        .map(|block| coefficient_of_variation(block))
        .collect::<Result<Vec<_>, _>>()
        .map(|values| values.into_iter().max_by(f64::total_cmp).unwrap())
}
fn between_block_cv(candidate: &CandidatePanel) -> Result<f64, ResearchError> {
    coefficient_of_variation(
        &candidate
            .blocks
            .iter()
            .map(|block| median(block))
            .collect::<Result<Vec<_>, _>>()?,
    )
}

fn load_frozen_pairs(path: &Path) -> Result<BTreeMap<PairedKey, f64>, ResearchError> {
    let text = fs::read_to_string(path)?;
    let mut out = BTreeMap::new();
    for line in text.lines().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() < 9 {
            return Err(ResearchError::Invalid("invalid frozen paired CSV"));
        }
        let key = (
            columns[0].to_string(),
            [
                parse_usize(columns[2])?,
                parse_usize(columns[3])?,
                parse_usize(columns[4])?,
            ],
            parse_usize(columns[5])?,
        );
        let value = columns[8]
            .parse::<f64>()
            .map_err(|_| ResearchError::Invalid("invalid frozen paired value"))?;
        if out.insert(key, value).is_some() {
            return Err(ResearchError::Invalid("duplicate frozen paired row"));
        }
    }
    if out.len() != 20 {
        return Err(ResearchError::Invalid(
            "frozen M7 paired output must contain 20 effects",
        ));
    }
    Ok(out)
}
fn load_frozen_bests(path: &Path) -> Result<BTreeMap<BestKey, Vec<String>>, ResearchError> {
    let text = fs::read_to_string(path)?;
    let mut out: BTreeMap<(String, [usize; 3]), Vec<String>> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let c = line.split(',').collect::<Vec<_>>();
        if c.len() < 16 {
            return Err(ResearchError::Invalid("invalid frozen reference CSV"));
        }
        if c[15] == "true" {
            out.entry((
                c[0].to_string(),
                [parse_usize(c[2])?, parse_usize(c[3])?, parse_usize(c[4])?],
            ))
            .or_default()
            .push(c[5].to_string());
        }
    }
    for values in out.values_mut() {
        values.sort();
    }
    if out.len() != 10 {
        return Err(ResearchError::Invalid(
            "frozen M7 reference output must contain ten panels",
        ));
    }
    Ok(out)
}
fn parse_usize(value: &str) -> Result<usize, ResearchError> {
    value
        .parse()
        .map_err(|_| ResearchError::Invalid("invalid integer in frozen CSV"))
}
fn crosscheck_best(
    state: &ProspectiveSessionRecord,
    workload: [usize; 3],
    best: &[String],
    frozen: &BTreeMap<BestKey, Vec<String>>,
) -> Result<(), ResearchError> {
    if frozen
        .get(&(state.session_id.clone(), workload))
        .map(Vec::as_slice)
        != Some(best)
    {
        return Err(ResearchError::Invalid(
            "M8 P30 best set does not reproduce frozen M7",
        ));
    }
    Ok(())
}

fn verify_input_manifest(
    input: &Path,
    canonical: &Path,
) -> Result<(String, usize, String), ResearchError> {
    let expected = fs::read_to_string(canonical)?;
    let mut paths = Vec::new();
    for session in find_named(input, "session.json")? {
        collect_files(
            session
                .parent()
                .ok_or(ResearchError::Invalid("invalid session root"))?,
            &mut paths,
        )?;
    }
    paths.sort();
    paths.dedup();
    let mut generated = String::new();
    for path in &paths {
        let relative = path
            .strip_prefix(input)
            .map_err(|_| ResearchError::Invalid("input file escapes root"))?;
        generated.push_str(&format!(
            "{:x}  {}\n",
            Sha256::digest(fs::read(path)?),
            relative.to_string_lossy().replace('\\', "/")
        ));
    }
    let generated = generated.trim_end().to_string();
    if generated != expected.trim_end() {
        return Err(ResearchError::Invalid(
            "M7 input manifest verification failed",
        ));
    }
    Ok((
        generated,
        paths.len(),
        format!("{:x}", Sha256::digest(expected.as_bytes())),
    ))
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

fn session_summary_csv(rows: &[SessionSummaryRow]) -> String {
    let mut csv = String::from(
        "schema,session_index,m,k,n,budget,subset_sign_changes,worst_absolute_paired_change,p30_paired_effect,statistical_unit\n",
    );
    for (session, w, budget, changes, worst, p30) in rows {
        csv.push_str(&format!(
            "{OUTPUT_SCHEMA},{session},{},{},{},{budget},{changes},{worst},{p30},session\n",
            w[0], w[1], w[2]
        ));
    }
    csv
}
fn report(analysis: &ReferenceReliabilityAnalysis, rows: &[SessionSummaryRow]) -> String {
    let sign_changes = rows.iter().filter(|row| row.3 > 0).count();
    let worst = rows
        .iter()
        .map(|row| row.4)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    format!(
        "# M8 reference measurement reliability\n\n**RETROSPECTIVE / EXPLORATORY.** This analysis plan was frozen before this M8 procedure was run, but after the M7 data were collected and inspected. It is not confirmatory inference or independent GPU validation.\n\nValidated {} sessions, {} workload panels, {} candidate references, {} blocks, and {} stored timing samples. The full P30 estimator reproduced all frozen M7 empirical-best sets and 20 paired policy effects.\n\nStored-order P10/P20 prefixes and three shared leave-one-block-out panels were compared with P30. P30 is a measured full panel, not oracle truth. Across the 20 session/condition contrasts, {} had at least one subset sign disagreement with P30; the largest absolute paired-effect change was {}. See the CSV artifacts for every raw session-level diagnostic.\n\nThe independent unit remains the five M7 sessions. Prefixes and LOBO subsets reuse observations and are dependent. No timestamp, block, workload, or contrast is promoted to an independent session. All extreme observations were retained. The records cannot identify thermal, power, driver, or background-load causes; establish a true-best kernel; validate a sample-size optimum; support population superiority; or generalize beyond this Apple M5/Metal dataset.\n",
        analysis.sessions,
        analysis.panels,
        analysis.candidates,
        analysis.reference_blocks,
        analysis.reference_samples,
        sign_changes,
        worst
    )
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
        manifest.push_str(&format!(
            "{:x}  {}\n",
            Sha256::digest(fs::read(&path)?),
            path.file_name().unwrap().to_string_lossy()
        ));
    }
    write_new(&output.join("SHA256SUMS"), manifest.trim_end().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidate(id: &str, blocks: [[f64; 10]; 3]) -> CandidatePanel {
        CandidatePanel {
            id: id.into(),
            blocks: blocks.map(Vec::from),
        }
    }
    fn constant(value: f64) -> [[f64; 10]; 3] {
        [[value; 10], [value; 10], [value; 10]]
    }
    #[test]
    fn median_even_and_invalid() {
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert!(median(&[0.0]).is_err());
        assert!(median(&[f64::NAN]).is_err());
    }
    #[test]
    fn prefixes_and_lobo_use_stored_membership() {
        let mut b = constant(1.0);
        b[1] = [2.0; 10];
        b[2] = [3.0; 10];
        let panels = subset_panels(&[candidate("a", b)]).unwrap();
        assert_eq!(
            panels.iter().map(|p| p.values[0].1).collect::<Vec<_>>(),
            [1.0, 1.5, 2.0, 2.5, 2.0, 1.5]
        );
    }
    #[test]
    fn stable_ranking_ties_and_prefix_reversal() {
        let mut a = constant(2.0);
        a[0] = [1.0; 10];
        let panels = subset_panels(&[candidate("a", a), candidate("b", constant(1.5))]).unwrap();
        assert_eq!(panels[0].best, ["a"]);
        assert_eq!(panels[2].best, ["b"]);
        assert_eq!(average_ranks(&[1.0, 1.0, 2.0]), [1.5, 1.5, 3.0]);
    }
    #[test]
    fn block_omission_can_reverse_ranking() {
        let mut a = constant(1.0);
        a[0] = [100.0; 10];
        let mut b = constant(2.0);
        b[0] = [1.0; 10];
        let p = subset_panels(&[candidate("a", a), candidate("b", b)]).unwrap();
        assert_eq!(p[3].best, ["a"]);
        assert_eq!(p[4].best, ["b"]);
    }
    #[test]
    fn individual_block_panel_preserves_candidate_coverage() {
        let candidates = [candidate("a", constant(1.0)), candidate("b", constant(2.0))];
        let block = block_panel(&candidates, 1).unwrap();
        assert_eq!(block.name, "BLOCK-1");
        assert_eq!(block.best, ["a"]);
        assert_eq!(block.ranks, [1.0, 2.0]);
        assert!(block_panel(&candidates, 3).is_err());
    }
    #[test]
    fn shared_denominator_regret_invariant_and_equal_selection() {
        let p =
            subset_panels(&[candidate("a", constant(1.0)), candidate("b", constant(2.0))]).unwrap();
        assert_eq!(regrets(&p[2], "a", "a").unwrap(), (0.0, 0.0, 0.0));
        assert_eq!(regrets(&p[2], "b", "a").unwrap().2, 1.0);
    }
    #[test]
    fn extreme_outlier_is_retained() {
        let mut a = constant(1.0);
        a[0][0] = 1_000_000.0;
        assert!(within_block_cv_max(&candidate("a", a)).unwrap() > 2.0);
    }
    #[test]
    fn undefined_rank_metric_is_explicit() {
        let values = vec![("a".into(), 1.0), ("b".into(), 1.0)];
        assert_eq!(rank_agreement(&values, &values), (None, None));
    }
    #[test]
    fn plan_paths_reject_escape() {
        assert!(safe_relative(Path::new("results/x")));
        assert!(!safe_relative(Path::new("../x")));
        assert!(!safe_relative(Path::new("/x")));
    }
}

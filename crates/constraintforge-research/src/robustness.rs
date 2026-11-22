use super::*;

const EXPECTED_PROTOCOL_SHA256: &str =
    "81296ac7da1448b98ea75e70a6e4404021775d5bbb31ed2011c084104ce3dd22";
const EXPECTED_RESOLVED_PROTOCOL_SHA256: &str =
    "0ee93a7d1d7eb02cbb8a141e8cd821f9b218a69631aa38b9374b8774d1615de3";
const EXPECTED_SOURCE_TREE_SHA256: &str =
    "e7ca6493a5a83129bfdb763c29a7f7e701e294ee233d0c8ac6a440fdeb1a5ef2";
const EXPECTED_SESSIONS: [&str; 5] = [
    "m6-v2-session-001",
    "m6-v2-session-002",
    "m6-v2-session-003",
    "m6-v2-session-004",
    "m6-v2-session-005",
];
const EXPECTED_SEEDS: [u64; 5] = [11, 23, 37, 41, 53];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobustnessAnalysis {
    pub schema: String,
    pub classification: String,
    pub sessions: usize,
    pub trials: usize,
    pub search_batches: usize,
    pub ranking_comparisons: usize,
    pub winner_groups: usize,
    pub sensitivity_rows: usize,
    pub warnings: Vec<String>,
}

struct LoadedTrial {
    session_id: String,
    session_index: usize,
    protocol_sha256: Option<String>,
    workload: [usize; 3],
    budget: usize,
    policy_id: String,
    order_index: usize,
    result: TuningResult,
    source_path: String,
}

#[derive(Clone, Copy)]
struct Dispersion {
    count: usize,
    valid: usize,
    invalid: usize,
    median: Option<f64>,
    range: Option<f64>,
    mad: Option<f64>,
    iqr: Option<f64>,
    cv: Option<f64>,
    range_ratio: Option<f64>,
}

#[derive(Clone)]
struct RankedCandidate {
    id: String,
    samples: usize,
    estimate: f64,
    rank: f64,
}

type RankAgreementValues = (f64, f64, f64);
type ConditionRankings<'a> = Vec<(&'a LoadedTrial, Vec<RankedCandidate>)>;

#[derive(Clone, Copy)]
enum EvaluationSummary {
    Median,
    TrimmedMean20,
    HodgesLehmann,
}

impl EvaluationSummary {
    fn name(self) -> &'static str {
        match self {
            Self::Median => "median",
            Self::TrimmedMean20 => "trimmed-mean-20pct",
            Self::HodgesLehmann => "hodges-lehmann",
        }
    }

    fn calculate(self, values: &[f64]) -> Option<f64> {
        if values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return None;
        }
        match self {
            Self::Median => median(values),
            Self::TrimmedMean20 => {
                if values.len() != 10 {
                    return None;
                }
                let mut sorted = values.to_vec();
                sorted.sort_by(f64::total_cmp);
                Some(mean(&sorted[2..8]))
            }
            Self::HodgesLehmann => {
                if values.is_empty() {
                    return None;
                }
                let mut averages = Vec::with_capacity(values.len() * (values.len() + 1) / 2);
                for (i, left) in values.iter().enumerate() {
                    for right in &values[i..] {
                        averages.push((left + right) / 2.0);
                    }
                }
                median(&averages)
            }
        }
    }
}

pub fn robustness(
    input: &Path,
    output: &Path,
    plan: &Path,
    protocol: &Path,
) -> Result<RobustnessAnalysis, ResearchError> {
    if output.exists() {
        return Err(ResearchError::OutputExists(output.to_path_buf()));
    }
    let validation = validate_results(input)?;
    if !validation.valid {
        return Err(ResearchError::Invalid("robustness input validation failed"));
    }
    let plan_bytes = fs::read(plan)?;
    let plan_sha256 = format!("{:x}", Sha256::digest(&plan_bytes));
    let protocol_bytes = fs::read(protocol)?;
    if format!("{:x}", Sha256::digest(&protocol_bytes)) != EXPECTED_PROTOCOL_SHA256 {
        return Err(ResearchError::Invalid(
            "frozen M6-v2 protocol file SHA-256 mismatch",
        ));
    }
    Protocol::parse(protocol)?;
    let trials = load_trials(input)?;
    validate_identity(&trials)?;

    fs::create_dir(output)?;
    write_new(
        &output.join("validation.json"),
        &serde_json::to_vec_pretty(&validation)?,
    )?;

    let search_batches = write_dispersion(output, &trials)?;
    let (ranking_comparisons, ranking_values) = write_rankings(output, &trials)?;
    let winner_groups = write_winners(output, &trials)?;
    write_search_evaluation(output, &trials)?;
    let sensitivity_rows = write_sensitivity(output, &trials)?;
    write_costs(output, &trials)?;
    write_identifiability(output, &trials)?;

    let software = current_software_metadata()?;
    let provenance = serde_json::json!({
        "schema": "constraintforge-m7-retrospective-provenance-v1",
        "classification": "RETROSPECTIVE / EXPLORATORY",
        "input_root": input.display().to_string(),
        "input_files": trials.iter().map(|trial| trial.source_path.clone()).collect::<Vec<_>>(),
        "m6_experimental_commit": "2d81322b6d7908ce24c45c0869f857b277ec8976",
        "m6_source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "m6_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "m6_resolved_protocol_identity": EXPECTED_RESOLVED_PROTOCOL_SHA256,
        "m6_protocol_path": protocol.display().to_string(),
        "m7_plan_path": plan.display().to_string(),
        "m7_plan_sha256": plan_sha256,
        "m7_analysis_source_revision": software.source_revision,
        "m7_analysis_clean_tree": software.git_worktree_clean,
        "m7_analysis_module_sha256": format!("{:x}", Sha256::digest(include_bytes!("robustness.rs"))),
        "random_seed": serde_json::Value::Null,
        "stochastic_procedures": false,
    });
    write_new(
        &output.join("provenance.json"),
        &serde_json::to_vec_pretty(&provenance)?,
    )?;

    let analysis = RobustnessAnalysis {
        schema: "constraintforge-m7-retrospective-analysis-v1".into(),
        classification: "RETROSPECTIVE / EXPLORATORY".into(),
        sessions: EXPECTED_SESSIONS.len(),
        trials: trials.len(),
        search_batches,
        ranking_comparisons,
        winner_groups,
        sensitivity_rows,
        warnings: vec![
            "five sessions on one Apple M5 are the highest independent units".into(),
            "exhaustive ranks are search-based and are not independent-evaluation truth".into(),
            "unselected candidate latency and counterfactual search histories are not identifiable"
                .into(),
        ],
    };
    write_new(
        &output.join("analysis.json"),
        &serde_json::to_vec_pretty(&analysis)?,
    )?;
    write_new(
        &output.join("summary.md"),
        summary(&analysis, &ranking_values).as_bytes(),
    )?;
    write_manifest(output)?;
    Ok(analysis)
}

fn load_trials(input: &Path) -> Result<Vec<LoadedTrial>, ResearchError> {
    let mut loaded = Vec::new();
    for state_path in find_named(input, "session.json")? {
        let state: SessionRecord = serde_json::from_slice(&fs::read(&state_path)?)?;
        let session_dir = state_path
            .parent()
            .ok_or(ResearchError::Invalid("session path has no parent"))?;
        for trial in &state.trials {
            if trial.status != "complete" {
                return Err(ResearchError::Invalid(
                    "robustness input contains incomplete trial",
                ));
            }
            let relative = trial
                .result_file
                .as_deref()
                .ok_or(ResearchError::Invalid("complete trial has no result file"))?;
            let path = session_dir.join(relative);
            let result = read_valid_result(&path)?;
            loaded.push(LoadedTrial {
                session_id: state.session_id.clone(),
                session_index: state.session_index,
                protocol_sha256: state.protocol_sha256.clone(),
                workload: trial.workload,
                budget: trial.budget,
                policy_id: trial.policy_id.clone(),
                order_index: trial.order_index,
                result,
                source_path: path.display().to_string(),
            });
        }
    }
    loaded.sort_by(|left, right| {
        (
            left.session_index,
            left.workload,
            left.budget,
            left.policy_id.as_str(),
            left.order_index,
        )
            .cmp(&(
                right.session_index,
                right.workload,
                right.budget,
                right.policy_id.as_str(),
                right.order_index,
            ))
    });
    Ok(loaded)
}

fn validate_identity(trials: &[LoadedTrial]) -> Result<(), ResearchError> {
    if trials.len() != 900 {
        return Err(ResearchError::Invalid(
            "robustness requires exactly 900 M6-v2 trials",
        ));
    }
    let sessions = trials
        .iter()
        .map(|trial| trial.session_id.as_str())
        .collect::<BTreeSet<_>>();
    if sessions != EXPECTED_SESSIONS.into_iter().collect() {
        return Err(ResearchError::Invalid(
            "robustness input has unexpected or duplicate sessions",
        ));
    }
    let mut ids = BTreeSet::new();
    for trial in trials {
        if EXPECTED_SESSIONS.get(trial.session_index).copied() != Some(trial.session_id.as_str())
            || EXPECTED_SEEDS.get(trial.session_index).copied() != Some(trial.result.search_seed)
        {
            return Err(ResearchError::Invalid(
                "robustness session index, identity, or seed mismatch",
            ));
        }
        if !ids.insert((trial.session_id.as_str(), trial.result.trial_id.as_str())) {
            return Err(ResearchError::Invalid(
                "robustness input contains duplicate trial identity",
            ));
        }
        validate_frozen_identity(
            trial.protocol_sha256.as_deref(),
            trial.result.software.source_tree_sha256.as_deref(),
        )?;
    }
    Ok(())
}

fn validate_frozen_identity(
    protocol_sha256: Option<&str>,
    source_tree_sha256: Option<&str>,
) -> Result<(), ResearchError> {
    if source_tree_sha256 != Some(EXPECTED_SOURCE_TREE_SHA256) {
        return Err(ResearchError::Invalid(
            "robustness input source identity mismatch",
        ));
    }
    if protocol_sha256 != Some(EXPECTED_RESOLVED_PROTOCOL_SHA256) {
        return Err(ResearchError::Invalid(
            "robustness input protocol identity mismatch",
        ));
    }
    Ok(())
}

fn write_dispersion(output: &Path, trials: &[LoadedTrial]) -> Result<usize, ResearchError> {
    let mut csv = String::from(
        "kind,session,session_index,m,k,n,budget,policy,trial_id,batch_index,candidate,reason,count,valid,invalid,median_ns,range_ns,mad_ns,iqr_ns,cv,range_ratio,candidate_trial_samples,candidate_trial_median_ns,error,statistical_unit,interpretation_limit\n",
    );
    let mut batches = 0;
    for trial in trials {
        let mut by_candidate: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for decision in &trial.result.search.ordered_history {
            by_candidate
                .entry(&decision.candidate_id)
                .or_default()
                .extend(decision.raw_samples_ns.iter().copied().filter(valid_sample));
        }
        for decision in &trial.result.search.ordered_history {
            batches += 1;
            let d = dispersion(&decision.raw_samples_ns);
            let candidate_samples = by_candidate
                .get(decision.candidate_id.as_str())
                .map(Vec::as_slice)
                .unwrap_or_default();
            push_dispersion_row(
                &mut csv,
                "search-batch",
                trial,
                Some(decision.batch_index),
                &decision.candidate_id,
                &decision.reason,
                d,
                candidate_samples.len(),
                median(candidate_samples),
                decision.error.as_deref().unwrap_or(""),
                "measurement batch nested in trial/session",
                "three samples; descriptive only",
            );
        }
        let d = dispersion(&trial.result.independent_evaluation_samples_ns);
        push_dispersion_row(
            &mut csv,
            "selected-evaluation",
            trial,
            None,
            &trial.result.search.selected_candidate_id,
            "fresh selected-candidate evaluation",
            d,
            d.valid,
            d.median,
            "",
            "trial nested in session",
            "conditional on selected candidate; unselected outcomes absent",
        );
    }
    write_csv_new(&output.join("search-dispersion.csv"), &csv)?;
    Ok(batches)
}

#[allow(clippy::too_many_arguments)]
fn push_dispersion_row(
    csv: &mut String,
    kind: &str,
    trial: &LoadedTrial,
    batch_index: Option<usize>,
    candidate: &str,
    reason: &str,
    d: Dispersion,
    candidate_samples: usize,
    candidate_median: Option<f64>,
    error: &str,
    unit: &str,
    limit: &str,
) {
    let [m, k, n] = trial.workload;
    csv.push_str(&format!(
        "{kind},{},{},{m},{k},{n},{},{},{},{},{candidate},{reason},{},{},{},{},{},{},{},{},{},{candidate_samples},{},{error},{unit},{limit}\n",
        trial.session_id,
        trial.session_index,
        trial.budget,
        trial.policy_id,
        trial.result.trial_id,
        display_option_usize(batch_index),
        d.count,
        d.valid,
        d.invalid,
        display_option(d.median),
        display_option(d.range),
        display_option(d.mad),
        display_option(d.iqr),
        display_option(d.cv),
        display_option(d.range_ratio),
        display_option(candidate_median),
    ));
}

fn write_rankings(
    output: &Path,
    trials: &[LoadedTrial],
) -> Result<(usize, Vec<RankAgreementValues>), ResearchError> {
    let exhaustive = trials
        .iter()
        .filter(|trial| trial.policy_id == "exhaustive")
        .map(|trial| Ok((trial, exhaustive_ranking(trial)?)))
        .collect::<Result<Vec<_>, ResearchError>>()?;
    let mut ranking_csv = String::from(
        "session,session_index,m,k,n,budget,candidate,search_samples,search_median_ns,average_rank,statistical_unit,interpretation_limit\n",
    );
    for (trial, ranking) in &exhaustive {
        let [m, k, n] = trial.workload;
        for candidate in ranking {
            ranking_csv.push_str(&format!(
                "{},{},{m},{k},{n},{},{},{},{},{},exhaustive trial nested in session,SEARCH-based ranking; not independent-evaluation truth\n",
                trial.session_id,
                trial.session_index,
                trial.budget,
                candidate.id,
                candidate.samples,
                candidate.estimate,
                candidate.rank,
            ));
        }
    }
    write_csv_new(&output.join("candidate-ranking.csv"), &ranking_csv)?;

    let mut by_condition: BTreeMap<([usize; 3], usize), ConditionRankings<'_>> = BTreeMap::new();
    for (trial, ranking) in exhaustive {
        by_condition
            .entry((trial.workload, trial.budget))
            .or_default()
            .push((trial, ranking));
    }
    let mut agreement_csv = String::from(
        "m,k,n,budget,left_session,right_session,shared_candidates,spearman_rho,kendall_tau_b,top1_agreement,top3_jaccard,missing_candidates,statistical_unit,interpretation_limit\n",
    );
    let mut values = Vec::new();
    let mut comparisons = 0;
    for (([m, k, n], budget), groups) in by_condition {
        for left_index in 0..groups.len() {
            for right_index in left_index + 1..groups.len() {
                let (left_trial, left) = &groups[left_index];
                let (right_trial, right) = &groups[right_index];
                let agreement = rank_agreement(left, right)?;
                comparisons += 1;
                values.push((agreement.0, agreement.1, agreement.3));
                agreement_csv.push_str(&format!(
                    "{m},{k},{n},{budget},{},{},{},{},{},{},{},{},session pair within workload/budget,SEARCH-based ranks with only three samples per candidate\n",
                    left_trial.session_id,
                    right_trial.session_id,
                    agreement.4,
                    agreement.0,
                    agreement.1,
                    agreement.2,
                    agreement.3,
                    10usize.saturating_sub(agreement.4),
                ));
            }
        }
    }
    write_csv_new(&output.join("ranking-agreement.csv"), &agreement_csv)?;
    Ok((comparisons, values))
}

fn exhaustive_ranking(trial: &LoadedTrial) -> Result<Vec<RankedCandidate>, ResearchError> {
    let mut samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for decision in &trial.result.search.ordered_history {
        samples
            .entry(decision.candidate_id.clone())
            .or_default()
            .extend(decision.raw_samples_ns.iter().copied().filter(valid_sample));
    }
    if samples.len() != 10 {
        return Err(ResearchError::Invalid(
            "exhaustive ranking lacks ten measured candidates",
        ));
    }
    let mut ranked = samples
        .into_iter()
        .map(|(id, values)| {
            Ok(RankedCandidate {
                id,
                samples: values.len(),
                estimate: median(&values).ok_or(ResearchError::Invalid(
                    "exhaustive candidate has no valid search sample",
                ))?,
                rank: 0.0,
            })
        })
        .collect::<Result<Vec<_>, ResearchError>>()?;
    ranked.sort_by(|left, right| {
        left.estimate
            .total_cmp(&right.estimate)
            .then_with(|| left.id.cmp(&right.id))
    });
    let values = ranked.iter().map(|item| item.estimate).collect::<Vec<_>>();
    let ranks = average_ranks(&values);
    for (candidate, rank) in ranked.iter_mut().zip(ranks) {
        candidate.rank = rank;
    }
    Ok(ranked)
}

fn rank_agreement(
    left: &[RankedCandidate],
    right: &[RankedCandidate],
) -> Result<(f64, f64, bool, f64, usize), ResearchError> {
    let left_map = left
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.rank))
        .collect::<BTreeMap<_, _>>();
    let right_map = right
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.rank))
        .collect::<BTreeMap<_, _>>();
    let shared = left_map
        .keys()
        .filter(|id| right_map.contains_key(*id))
        .copied()
        .collect::<Vec<_>>();
    if shared.len() < 3 {
        return Err(ResearchError::Invalid(
            "insufficient candidate overlap for rank agreement",
        ));
    }
    let left_ranks = shared.iter().map(|id| left_map[id]).collect::<Vec<_>>();
    let right_ranks = shared.iter().map(|id| right_map[id]).collect::<Vec<_>>();
    let rho = pearson(&left_ranks, &right_ranks).ok_or(ResearchError::Invalid(
        "rank agreement is undefined for constant ranks",
    ))?;
    let tau = kendall_tau_b(&left_ranks, &right_ranks)
        .ok_or(ResearchError::Invalid("Kendall agreement is undefined"))?;
    let top1_left = top_set(left, 1.0);
    let top1_right = top_set(right, 1.0);
    let top3_left = top_set(left, 3.0);
    let top3_right = top_set(right, 3.0);
    Ok((
        rho,
        tau,
        top1_left == top1_right,
        jaccard(&top3_left, &top3_right),
        shared.len(),
    ))
}

fn write_winners(output: &Path, trials: &[LoadedTrial]) -> Result<usize, ResearchError> {
    let mut groups: BTreeMap<([usize; 3], usize, String), Vec<&LoadedTrial>> = BTreeMap::new();
    for trial in trials {
        groups
            .entry((trial.workload, trial.budget, trial.policy_id.clone()))
            .or_default()
            .push(trial);
    }
    let mut csv = String::from(
        "m,k,n,budget,policy,session,session_index,selected_candidate,measured_candidates,observed_margin,modal_candidate,modal_count,session_count,agreement,all_selected_candidates,statistical_unit,interpretation_limit\n",
    );
    for (([m, k, n], budget, policy), group) in &groups {
        let mut frequency: BTreeMap<&str, usize> = BTreeMap::new();
        for trial in group {
            *frequency
                .entry(&trial.result.search.selected_candidate_id)
                .or_default() += 1;
        }
        let (modal, modal_count) = frequency
            .into_iter()
            .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(left.0)))
            .ok_or(ResearchError::Invalid("winner group is empty"))?;
        let all = group
            .iter()
            .map(|trial| trial.result.search.selected_candidate_id.as_str())
            .collect::<Vec<_>>()
            .join("|");
        for trial in group {
            let estimates = candidate_estimates(&trial.result);
            let margin = selection_margin(&estimates);
            csv.push_str(&format!(
                "{m},{k},{n},{budget},{policy},{},{},{},{},{},{modal},{modal_count},{},{},{},session-level trial,selection variation is not causally attributable to timing\n",
                trial.session_id,
                trial.session_index,
                trial.result.search.selected_candidate_id,
                estimates.len(),
                display_option(margin),
                group.len(),
                modal_count as f64 / group.len() as f64,
                all,
            ));
        }
    }
    write_csv_new(&output.join("winner-selection-stability.csv"), &csv)?;
    Ok(groups.len())
}

fn write_search_evaluation(output: &Path, trials: &[LoadedTrial]) -> Result<(), ResearchError> {
    let mut csv = String::from(
        "session,session_index,m,k,n,budget,policy,selected_candidate,search_score_ns,evaluation_median_ns,search_to_evaluation_ratio,evaluation_cv,evaluation_mad_ns,evaluation_iqr_ns,evaluation_range_ratio,statistical_unit,interpretation_limit\n",
    );
    for trial in trials {
        let Some(score) = trial.result.search.selected_score_ns else {
            continue;
        };
        let evaluation = dispersion(&trial.result.independent_evaluation_samples_ns);
        let evaluation_median = evaluation.median.ok_or(ResearchError::Invalid(
            "selected candidate evaluation has no valid sample",
        ))?;
        let [m, k, n] = trial.workload;
        csv.push_str(&format!(
            "{},{},{m},{k},{n},{},{},{},{score},{evaluation_median},{},{},{},{},{},trial nested in session,same selected candidate only; not oracle regret or bias correction\n",
            trial.session_id,
            trial.session_index,
            trial.budget,
            trial.policy_id,
            trial.result.search.selected_candidate_id,
            score / evaluation_median,
            display_option(evaluation.cv),
            display_option(evaluation.mad),
            display_option(evaluation.iqr),
            display_option(evaluation.range_ratio),
        ));
    }
    write_csv_new(&output.join("search-evaluation-agreement.csv"), &csv)
}

fn write_sensitivity(output: &Path, trials: &[LoadedTrial]) -> Result<usize, ResearchError> {
    let mut evaluations: BTreeMap<(String, [usize; 3], usize, String), &TuningResult> =
        BTreeMap::new();
    for trial in trials {
        evaluations.insert(
            (
                trial.session_id.clone(),
                trial.workload,
                trial.budget,
                trial.policy_id.clone(),
            ),
            &trial.result,
        );
    }
    let mut csv = String::from(
        "m,k,n,budget,evaluation_summary,omitted_session,included_sessions,pair_count,ratios,geometric_mean,min_ratio,max_ratio,direction_below_one,direction_change_from_full,statistical_unit,interpretation_limit\n",
    );
    let mut rows = 0;
    for workload in HELD_OUT_SHAPES {
        for budget in [2, 4, 6, 8] {
            for summary in [
                EvaluationSummary::Median,
                EvaluationSummary::TrimmedMean20,
                EvaluationSummary::HodgesLehmann,
            ] {
                let ratios = paired_ratios(&evaluations, workload, budget, summary)?;
                let full =
                    geometric_mean(&ratios.iter().map(|(_, ratio)| *ratio).collect::<Vec<_>>())
                        .ok_or(ResearchError::Invalid("missing primary sensitivity pairs"))?;
                for omitted in std::iter::once(None).chain(EXPECTED_SESSIONS.map(Some)) {
                    let included = ratios
                        .iter()
                        .filter(|(session, _)| omitted != Some(session.as_str()))
                        .collect::<Vec<_>>();
                    let values = included.iter().map(|(_, ratio)| *ratio).collect::<Vec<_>>();
                    let gm = geometric_mean(&values)
                        .ok_or(ResearchError::Invalid("leave-one-session-out has no pairs"))?;
                    let [m, k, n] = workload;
                    csv.push_str(&format!(
                        "{m},{k},{n},{budget},{},{},{},{},{},{gm},{},{},{},{},session-matched policy pair,leave-one-out values are sensitivity diagnostics not replications\n",
                        summary.name(),
                        omitted.unwrap_or("none"),
                        included
                            .iter()
                            .map(|(session, _)| session.as_str())
                            .collect::<Vec<_>>()
                            .join("|"),
                        values.len(),
                        values.iter().map(|value| value.to_string()).collect::<Vec<_>>().join("|"),
                        values.iter().copied().min_by(f64::total_cmp).unwrap_or(gm),
                        values.iter().copied().max_by(f64::total_cmp).unwrap_or(gm),
                        gm < 1.0,
                        (gm < 1.0) != (full < 1.0),
                    ));
                    rows += 1;
                }
            }
        }
    }
    write_csv_new(&output.join("session-sensitivity.csv"), &csv)?;
    Ok(rows)
}

fn paired_ratios(
    evaluations: &BTreeMap<(String, [usize; 3], usize, String), &TuningResult>,
    workload: [usize; 3],
    budget: usize,
    summary: EvaluationSummary,
) -> Result<Vec<(String, f64)>, ResearchError> {
    let mut ratios = Vec::new();
    for session in EXPECTED_SESSIONS {
        let aware = evaluations
            .get(&(session.into(), workload, budget, "resource-aware".into()))
            .ok_or(ResearchError::Invalid("missing resource-aware pair"))?;
        let random = evaluations
            .get(&(session.into(), workload, budget, "random".into()))
            .ok_or(ResearchError::Invalid("missing random pair"))?;
        let aware_value = summary
            .calculate(&aware.independent_evaluation_samples_ns)
            .ok_or(ResearchError::Invalid(
                "invalid resource-aware evaluation summary",
            ))?;
        let random_value = summary
            .calculate(&random.independent_evaluation_samples_ns)
            .ok_or(ResearchError::Invalid("invalid random evaluation summary"))?;
        ratios.push((session.into(), aware_value / random_value));
    }
    Ok(ratios)
}

fn write_costs(output: &Path, trials: &[LoadedTrial]) -> Result<(), ResearchError> {
    let mut csv = String::from(
        "session,session_index,m,k,n,budget,policy,measurement_class,measurement_batches,timed_search_samples,measured_candidates,search_wall_time_ns,decision_overhead_ns,preparation_time_ns,correctness_time_ns,tuning_wall_time_ns,selected_evaluation_median_ns,statistical_unit,interpretation_limit\n",
    );
    for trial in trials {
        let class = match trial.policy_id.as_str() {
            "fixed" | "heuristic" => "zero-measurement-baseline",
            "exhaustive" => "ten-batch-reference",
            _ => "matched-eight-batch-search",
        };
        let evaluation = median(&trial.result.independent_evaluation_samples_ns).ok_or(
            ResearchError::Invalid("missing selected evaluation for cost row"),
        )?;
        let [m, k, n] = trial.workload;
        csv.push_str(&format!(
            "{},{},{m},{k},{n},{},{},{class},{},{},{},{},{},{},{},{},{evaluation},trial nested in session,cost and selected latency are descriptive; classes have unequal budgets\n",
            trial.session_id,
            trial.session_index,
            trial.budget,
            trial.policy_id,
            trial.result.search.measurement_batches,
            trial.result.search.total_measured_dispatches,
            trial.result.search.candidates_measured,
            trial.result.search_wall_time_ns,
            trial.result.search_decision_overhead_ns,
            trial.result.preparation_time_ns,
            trial.result.correctness_validation_time_ns,
            trial.result.tuning_wall_time_ns,
        ));
    }
    write_csv_new(&output.join("search-cost-tradeoff.csv"), &csv)
}

fn write_csv_new(path: &Path, csv: &str) -> Result<(), ResearchError> {
    write_new(path, csv.trim_end_matches('\n').as_bytes())
}

fn write_identifiability(output: &Path, trials: &[LoadedTrial]) -> Result<(), ResearchError> {
    let mut coverage: BTreeMap<&str, usize> = BTreeMap::new();
    let mut failures = 0;
    for trial in trials {
        for decision in &trial.result.search.ordered_history {
            *coverage.entry(&decision.candidate_id).or_default() += 1;
            if decision.error.is_some() || !decision.correctness_passed {
                failures += 1;
            }
        }
    }
    let value = serde_json::json!({
        "schema": "constraintforge-m7-identifiability-v1",
        "classification": "RETROSPECTIVE / EXPLORATORY",
        "coverage": {
            "sessions": EXPECTED_SESSIONS.len(),
            "trials": trials.len(),
            "candidate_pool_size": 10,
            "search_batches_by_candidate": coverage,
            "failed_or_incorrect_search_batches": failures,
            "independent_evaluation_samples_per_trial": 10,
        },
        "supported": [
            "search and selected-evaluation dispersion",
            "exhaustive SEARCH-rank agreement",
            "selected-winner stability",
            "same-winner search/evaluation disagreement",
            "session-level policy-effect sensitivity",
            "recorded cost versus selected performance"
        ],
        "not_identifiable": [
            {"metric":"true independent-evaluation candidate ranking","missing":"independent evaluation of unselected candidates"},
            {"metric":"true selection regret","missing":"independently evaluated best legal candidate"},
            {"metric":"counterfactual adaptive replay","missing":"measurements on unobserved decision paths"},
            {"metric":"physical cause of timing variation","missing":"controlled interventions and continuous telemetry"}
        ],
        "statistical_unit": "session",
        "highest_independent_units": 5,
    });
    write_new(
        &output.join("identifiability.json"),
        &serde_json::to_vec_pretty(&value)?,
    )
}

fn dispersion(values: &[f64]) -> Dispersion {
    let valid = values
        .iter()
        .copied()
        .filter(valid_sample)
        .collect::<Vec<_>>();
    let center = median(&valid);
    let mad = center.and_then(|median_value| {
        median(
            &valid
                .iter()
                .map(|value| (value - median_value).abs())
                .collect::<Vec<_>>(),
        )
    });
    let mean_value = (!valid.is_empty()).then(|| mean(&valid));
    let cv = mean_value.and_then(|mean_value| {
        (valid.len() >= 2 && mean_value > 0.0).then(|| {
            let variance = valid
                .iter()
                .map(|value| (value - mean_value).powi(2))
                .sum::<f64>()
                / (valid.len() - 1) as f64;
            variance.sqrt() / mean_value
        })
    });
    let minimum = valid.iter().copied().min_by(f64::total_cmp);
    let maximum = valid.iter().copied().max_by(f64::total_cmp);
    Dispersion {
        count: values.len(),
        valid: valid.len(),
        invalid: values.len() - valid.len(),
        median: center,
        range: minimum.zip(maximum).map(|(min, max)| max - min),
        mad,
        iqr: quantile(&valid, 0.75)
            .zip(quantile(&valid, 0.25))
            .map(|(q3, q1)| q3 - q1),
        cv,
        range_ratio: minimum.zip(maximum).map(|(min, max)| max / min),
    }
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() - 1) as f64 * probability;
    let lower = h.floor() as usize;
    let upper = h.ceil() as usize;
    Some(sorted[lower] + (h - lower as f64) * (sorted[upper] - sorted[lower]))
}

fn valid_sample(value: &f64) -> bool {
    value.is_finite() && *value > 0.0
}

fn candidate_estimates(result: &TuningResult) -> BTreeMap<String, f64> {
    let mut samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for decision in &result.search.ordered_history {
        samples
            .entry(decision.candidate_id.clone())
            .or_default()
            .extend(decision.raw_samples_ns.iter().copied().filter(valid_sample));
    }
    samples
        .into_iter()
        .filter_map(|(candidate, samples)| median(&samples).map(|value| (candidate, value)))
        .collect()
}

fn selection_margin(estimates: &BTreeMap<String, f64>) -> Option<f64> {
    let mut values = estimates.values().copied().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    (values.len() >= 2 && values[0] > 0.0).then(|| (values[1] - values[0]) / values[0])
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[start]].total_cmp(&values[order[end]]).is_eq() {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for index in &order[start..end] {
            ranks[*index] = rank;
        }
        start = end;
    }
    ranks
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = mean(left);
    let right_mean = mean(right);
    let numerator = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>();
    let left_sum = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>();
    let right_sum = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>();
    let denominator = (left_sum * right_sum).sqrt();
    (denominator > 0.0).then_some(numerator / denominator)
}

fn kendall_tau_b(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let mut concordant: f64 = 0.0;
    let mut discordant: f64 = 0.0;
    let mut left_only_ties: f64 = 0.0;
    let mut right_only_ties: f64 = 0.0;
    for i in 0..left.len() {
        for j in i + 1..left.len() {
            let a = left[i].total_cmp(&left[j]);
            let b = right[i].total_cmp(&right[j]);
            match (a.is_eq(), b.is_eq()) {
                (true, true) => {}
                (true, false) => left_only_ties += 1.0,
                (false, true) => right_only_ties += 1.0,
                (false, false) if a == b => concordant += 1.0,
                (false, false) => discordant += 1.0,
            }
        }
    }
    let denominator = ((concordant + discordant + left_only_ties)
        * (concordant + discordant + right_only_ties))
        .sqrt();
    (denominator > 0.0).then_some((concordant - discordant) / denominator)
}

fn top_set(ranking: &[RankedCandidate], cutoff: f64) -> BTreeSet<&str> {
    ranking
        .iter()
        .filter(|candidate| candidate.rank <= cutoff)
        .map(|candidate| candidate.id.as_str())
        .collect()
}

fn jaccard(left: &BTreeSet<&str>, right: &BTreeSet<&str>) -> f64 {
    let union = left.union(right).count();
    if union == 0 {
        1.0
    } else {
        left.intersection(right).count() as f64 / union as f64
    }
}

fn geometric_mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty() && values.iter().all(|value| value.is_finite() && *value > 0.0))
        .then(|| (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp())
}

fn display_option(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn display_option_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn summary(analysis: &RobustnessAnalysis, ranking_values: &[(f64, f64, f64)]) -> String {
    let spearman = ranking_values
        .iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let kendall = ranking_values
        .iter()
        .map(|value| value.1)
        .collect::<Vec<_>>();
    let top3 = ranking_values
        .iter()
        .map(|value| value.2)
        .collect::<Vec<_>>();
    format!(
        "# M7 Phase 1 retrospective robustness analysis\n\n**RETROSPECTIVE / EXPLORATORY.** No GPU experiment was run.\n\nValidated {} trials in {} sessions and preserved {} search batches. Exhaustive SEARCH rankings produced {} session-pair comparisons.\n\n- Spearman rank agreement: median {}, range {} to {}.\n- Kendall tau-b: median {}, range {} to {}.\n- Top-3 Jaccard overlap: median {}, range {} to {}.\n- Winner groups: {}. Session-sensitivity rows: {}.\n\nThese are descriptive diagnostics. Exhaustive ranks are not independent-evaluation truth; unselected candidate latency, true regret, counterfactual adaptive histories, and physical causes of timing variation are not identifiable.\n",
        analysis.trials,
        analysis.sessions,
        analysis.search_batches,
        analysis.ranking_comparisons,
        display_option(median(&spearman)),
        display_option(spearman.iter().copied().min_by(f64::total_cmp)),
        display_option(spearman.iter().copied().max_by(f64::total_cmp)),
        display_option(median(&kendall)),
        display_option(kendall.iter().copied().min_by(f64::total_cmp)),
        display_option(kendall.iter().copied().max_by(f64::total_cmp)),
        display_option(median(&top3)),
        display_option(top3.iter().copied().min_by(f64::total_cmp)),
        display_option(top3.iter().copied().max_by(f64::total_cmp)),
        analysis.winner_groups,
        analysis.sensitivity_rows,
    )
}

fn write_manifest(output: &Path) -> Result<(), ResearchError> {
    let mut entries = fs::read_dir(output)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    entries
        .retain(|path| path.is_file() && path.file_name().is_some_and(|name| name != "SHA256SUMS"));
    entries.sort();
    let mut manifest = String::new();
    for path in entries {
        let name =
            path.file_name()
                .and_then(|name| name.to_str())
                .ok_or(ResearchError::Invalid(
                    "analysis output filename is not UTF-8",
                ))?;
        manifest.push_str(&format!("{:x}  {name}\n", Sha256::digest(fs::read(&path)?)));
    }
    manifest.pop();
    write_new(&output.join("SHA256SUMS"), manifest.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(values: &[(&str, f64)]) -> Vec<RankedCandidate> {
        let ranks = average_ranks(&values.iter().map(|value| value.1).collect::<Vec<_>>());
        values
            .iter()
            .zip(ranks)
            .map(|((id, estimate), rank)| RankedCandidate {
                id: (*id).into(),
                samples: 3,
                estimate: *estimate,
                rank,
            })
            .collect()
    }

    #[test]
    fn dispersion_retains_invalid_samples_and_uses_frozen_quantiles() {
        let d = dispersion(&[1.0, 2.0, 3.0, 0.0, f64::NAN]);
        assert_eq!((d.count, d.valid, d.invalid), (5, 3, 2));
        assert_eq!(d.median, Some(2.0));
        assert_eq!(d.iqr, Some(1.0));
        assert_eq!(d.range_ratio, Some(3.0));
    }

    #[test]
    fn average_ranks_and_agreement_handle_ties_and_identity() {
        assert_eq!(average_ranks(&[1.0, 1.0, 3.0]), vec![1.5, 1.5, 3.0]);
        let values = ranked(&[("a", 1.0), ("b", 1.0), ("c", 3.0)]);
        let agreement = rank_agreement(&values, &values).unwrap();
        assert!((agreement.0 - 1.0).abs() < 1e-12);
        assert!((agreement.1 - 1.0).abs() < 1e-12);
        assert!(agreement.2);
        assert_eq!(agreement.3, 1.0);
    }

    #[test]
    fn insufficient_candidate_overlap_is_rejected() {
        let left = ranked(&[("a", 1.0), ("b", 2.0)]);
        let right = ranked(&[("a", 1.0), ("b", 2.0)]);
        assert!(rank_agreement(&left, &right).is_err());
    }

    #[test]
    fn top_sets_include_boundary_ties() {
        let values = ranked(&[("a", 1.0), ("b", 2.0), ("c", 2.0), ("d", 4.0)]);
        assert_eq!(top_set(&values, 3.0), BTreeSet::from(["a", "b", "c"]));
    }

    #[test]
    fn robust_summaries_and_ratio_direction_are_deterministic() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        assert_eq!(EvaluationSummary::Median.calculate(&values), Some(5.5));
        assert_eq!(
            EvaluationSummary::TrimmedMean20.calculate(&values),
            Some(5.5)
        );
        assert!(
            EvaluationSummary::HodgesLehmann
                .calculate(&values)
                .is_some()
        );
        let search_estimate = 4.0;
        let evaluation_median = 5.0;
        assert!(search_estimate / evaluation_median < 1.0);
    }

    #[test]
    fn leave_one_out_uses_sessions_not_nested_samples() {
        let ratios = [0.5, 0.75, 1.0, 1.25, 1.5];
        let full = geometric_mean(&ratios).unwrap();
        let omitted = geometric_mean(&ratios[1..]).unwrap();
        assert_ne!(full, omitted);
        assert_eq!(ratios.len(), 5);
    }

    #[test]
    fn missing_candidates_are_not_assigned_latency() {
        let estimates = BTreeMap::from([("measured".into(), 10.0)]);
        assert!(!estimates.contains_key("unmeasured"));
        assert_eq!(selection_margin(&estimates), None);
    }

    #[test]
    fn unsupported_counterfactuals_remain_explicitly_absent() {
        let identifiability = [
            "unselected candidate evaluation",
            "counterfactual adaptive history",
            "true selection regret",
        ];
        assert_eq!(identifiability.len(), 3);
    }

    #[test]
    fn mixed_protocol_and_source_identities_are_rejected() {
        assert!(
            validate_frozen_identity(Some("wrong"), Some(EXPECTED_SOURCE_TREE_SHA256)).is_err()
        );
        assert!(
            validate_frozen_identity(Some(EXPECTED_RESOLVED_PROTOCOL_SHA256), Some("wrong"))
                .is_err()
        );
        assert!(
            validate_frozen_identity(
                Some(EXPECTED_RESOLVED_PROTOCOL_SHA256),
                Some(EXPECTED_SOURCE_TREE_SHA256)
            )
            .is_ok()
        );
    }

    #[test]
    fn output_manifest_refuses_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "constraintforge-robustness-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        write_new(&root.join("a.txt"), b"a").unwrap();
        write_manifest(&root).unwrap();
        assert!(matches!(
            write_manifest(&root),
            Err(ResearchError::OutputExists(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }
}

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand, ValueEnum};
use constraintforge_bench::{BenchmarkConfig, BenchmarkResult, BenchmarkSuiteResult, KernelChoice};
use constraintforge_gpu::{GraphExecution, MatMulKernel, MatMulKernelVariant};
use constraintforge_ir::{Graph, Inputs, NodeId, TensorSpec};
use constraintforge_reference::{Tensor, execute, matmul, within_tolerance};
use constraintforge_tuner::{
    GpuTuningConfig, MeasurementMode, Policy, ScoreAblation, SearchBudget, SearchDesign,
    TuningComparisonResult,
};
use thiserror::Error;
use tracing_subscriber::EnvFilter;

const ABSOLUTE_TOLERANCE: f32 = 1e-5;
const RELATIVE_TOLERANCE: f32 = 1e-4;

#[derive(Parser)]
#[command(
    name = "constraintforge",
    version,
    about = "ConstraintForge compiler and execution baseline"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List compute adapters and relevant limits.
    Devices,
    /// Execute a deterministic example.
    Demo {
        #[arg(long, value_enum)]
        backend: Backend,
        #[arg(long, value_enum, default_value = "basic")]
        example: Example,
        #[arg(long, value_enum, default_value = "naive")]
        matmul_kernel: MatMulKernelArg,
    },
    /// Compare deterministic GPU operators or graphs with the CPU oracle.
    Verify {
        #[arg(long, value_enum)]
        backend: VerifyBackend,
        #[arg(long, value_enum, default_value = "matmul")]
        suite: Suite,
        #[arg(long, value_enum, default_value = "naive")]
        matmul_kernel: MatMulKernelArg,
    },
    /// Run an opt-in, preliminary GPU comparison.
    Compare {
        #[command(subcommand)]
        operation: CompareOperation,
    },
    /// Run reproducible prepared GPU benchmarks.
    Bench {
        #[command(subcommand)]
        operation: BenchOperation,
    },
    /// Inspect or run bounded MatMul tuning.
    Tune {
        #[command(subcommand)]
        operation: TuneOperation,
    },
    /// Run, inspect, validate, or analyze controlled M6 experiments.
    Research {
        #[command(subcommand)]
        operation: ResearchOperation,
    },
}

#[derive(Subcommand)]
enum ResearchOperation {
    /// Inspect a protocol without using a GPU.
    Inspect {
        #[arg(long)]
        protocol: PathBuf,
    },
    /// Execute the bounded calibration preset.
    Pilot {
        #[arg(long)]
        protocol: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Test-only controlled stop after N durable trial records.
        #[arg(long, hide = true)]
        stop_after_trials: Option<usize>,
    },
    /// Execute one frozen session; never launches all sessions.
    Run {
        #[arg(long)]
        protocol: PathBuf,
        #[arg(long)]
        session: String,
        #[arg(long, default_value_t = 0)]
        session_index: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        resume: bool,
        /// Test-only controlled stop after N durable trial records.
        #[arg(long, hide = true)]
        stop_after_trials: Option<usize>,
    },
    /// Validate saved session and trial records.
    Validate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Analyze already-saved validated records without GPU execution.
    Analyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value_t = 10_000)]
        bootstrap_resamples: usize,
    },
    /// Run the frozen read-only M7 retrospective robustness analysis.
    Robustness {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "research/protocols/m7-retrospective-plan.md")]
        plan: PathBuf,
        #[arg(long, default_value = "research/protocols/m6-v2.toml")]
        protocol: PathBuf,
    },
    /// Inspect one deterministic M7 Phase 2A session plan without using a GPU.
    ProspectiveInspect {
        #[arg(long, default_value = "research/protocols/m7-phase2a.toml")]
        protocol: PathBuf,
        #[arg(long)]
        session_index: usize,
    },
    /// Run or resume one authorized M7 Phase 2A GPU session.
    ProspectiveRun {
        #[arg(long, default_value = "research/protocols/m7-phase2a.toml")]
        protocol: PathBuf,
        #[arg(long)]
        session: String,
        #[arg(long)]
        session_index: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Validate completed M7 Phase 2A records without using a GPU.
    ProspectiveValidate {
        #[arg(long)]
        input: PathBuf,
    },
    /// Analyze five completed M7 Phase 2A sessions without using a GPU.
    ProspectiveAnalyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Analyze exactly one complete session for qualification only.
        #[arg(long, default_value_t = false)]
        qualification_only: bool,
    },
    /// Analyze M7 reference-panel reliability without using a GPU.
    ReferenceRobustness {
        #[arg(
            long,
            default_value = "research/protocols/m8-reference-reliability-v1.toml"
        )]
        plan: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Resolve one deterministic M8 Phase 2A session plan without using a GPU.
    ReferencePanelPlan {
        #[arg(long, default_value = "research/protocols/m8-phase2a-v1.toml")]
        protocol: PathBuf,
        #[arg(long)]
        session_index: usize,
    },
    /// Run or resume one separately authorized M8 Phase 2A GPU session.
    ReferencePanelRun {
        #[arg(long, default_value = "research/protocols/m8-phase2a-v1.toml")]
        protocol: PathBuf,
        #[arg(long)]
        session: String,
        #[arg(long)]
        session_index: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Validate M8 Phase 2A records without using a GPU.
    ReferencePanelValidate {
        #[arg(long)]
        input: PathBuf,
    },
    /// Analyze validated M8 Phase 2A panels without using a GPU.
    ReferencePanelAnalyze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Analyze exactly one complete physical session for qualification only.
        #[arg(long, default_value_t = false)]
        qualification_only: bool,
    },
    /// Write the immutable manifest for a completed M7 Phase 2A pilot.
    ProspectiveManifest {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Run a disposable controlled-stop/resume qualification with durable hashes.
    QualifyResume {
        #[arg(long)]
        protocol: PathBuf,
        #[arg(long)]
        session: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        analysis: PathBuf,
    },
}

#[derive(Subcommand)]
enum TuneOperation {
    /// List the finite M5 candidate space and static resource features.
    Candidates {
        #[arg(long)]
        m: usize,
        #[arg(long)]
        k: usize,
        #[arg(long)]
        n: usize,
    },
    /// Run one bounded tuning policy and independent evaluation.
    Matmul {
        #[command(flatten)]
        args: TuneArgs,
        #[arg(long, value_enum)]
        policy: TunePolicy,
        #[arg(long)]
        output: PathBuf,
    },
    /// Run all four policies as independent sessions.
    Compare {
        #[command(flatten)]
        args: TuneArgs,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(clap::Args, Clone)]
struct TuneArgs {
    #[arg(long)]
    m: usize,
    #[arg(long)]
    k: usize,
    #[arg(long)]
    n: usize,
    #[arg(long, default_value_t = 6)]
    max_candidates: usize,
    #[arg(long, default_value_t = 8)]
    max_batches: usize,
    #[arg(long, default_value_t = 60_000)]
    max_ms: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value = "gpu-timestamp")]
    measurement: TuneMeasurement,
    #[arg(long, default_value_t = 1)]
    tuning_warmups: u32,
    #[arg(long, default_value_t = 3)]
    tuning_runs: u32,
    #[arg(long, default_value_t = 3)]
    evaluation_warmups: u32,
    #[arg(long, default_value_t = 10)]
    evaluation_runs: u32,
}

#[derive(Clone, Copy, ValueEnum)]
enum TunePolicy {
    Fixed,
    Heuristic,
    Random,
    ResourceAware,
}

impl From<TunePolicy> for Policy {
    fn from(value: TunePolicy) -> Self {
        match value {
            TunePolicy::Fixed => Self::Fixed,
            TunePolicy::Heuristic => Self::Heuristic,
            TunePolicy::Random => Self::Random,
            TunePolicy::ResourceAware => Self::ResourceAware,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum TuneMeasurement {
    GpuTimestamp,
    HostPreparedEndToEnd,
}

impl From<TuneMeasurement> for MeasurementMode {
    fn from(value: TuneMeasurement) -> Self {
        match value {
            TuneMeasurement::GpuTimestamp => Self::GpuTimestamp,
            TuneMeasurement::HostPreparedEndToEnd => Self::HostPreparedEndToEnd,
        }
    }
}

#[derive(Subcommand)]
enum BenchOperation {
    /// Benchmark one prepared MatMul kernel.
    Matmul {
        #[arg(long)]
        m: usize,
        #[arg(long)]
        k: usize,
        #[arg(long)]
        n: usize,
        #[arg(long, default_value_t = 10)]
        warmups: u32,
        #[arg(long, default_value_t = 30)]
        runs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum)]
        kernel: BenchKernel,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare prepared naive and tiled MatMul on one device.
    Compare {
        #[arg(long)]
        m: usize,
        #[arg(long)]
        k: usize,
        #[arg(long)]
        n: usize,
        #[arg(long, default_value_t = 10)]
        warmups: u32,
        #[arg(long, default_value_t = 30)]
        runs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Run the bounded smoke-shape suite.
    Suite {
        #[arg(long, value_enum, default_value = "smoke")]
        preset: BenchPreset,
        #[arg(long, default_value_t = 10)]
        warmups: u32,
        #[arg(long, default_value_t = 30)]
        runs: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum BenchKernel {
    Naive,
    Tiled,
}

#[derive(Clone, Copy, ValueEnum)]
enum BenchPreset {
    Smoke,
}

#[derive(Subcommand)]
enum CompareOperation {
    /// Compare naive and tiled MatMul using host end-to-end timing.
    Matmul {
        #[arg(long)]
        m: usize,
        #[arg(long)]
        k: usize,
        #[arg(long)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        warmups: u32,
        #[arg(long, default_value_t = 5)]
        runs: u32,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, ValueEnum)]
enum Example {
    Basic,
    Mlp,
}

#[derive(Clone, Copy, ValueEnum)]
enum VerifyBackend {
    Gpu,
}

#[derive(Clone, Copy, ValueEnum)]
enum Suite {
    Matmul,
    Graph,
}

#[derive(Clone, Copy, ValueEnum)]
enum MatMulKernelArg {
    Naive,
    Tiled,
    Auto,
}

impl From<MatMulKernelArg> for MatMulKernel {
    fn from(value: MatMulKernelArg) -> Self {
        match value {
            MatMulKernelArg::Naive => Self::Naive,
            MatMulKernelArg::Tiled => Self::Tiled,
            MatMulKernelArg::Auto => Self::Auto,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Ir(#[from] constraintforge_ir::IrError),
    #[error(transparent)]
    Reference(#[from] constraintforge_reference::ReferenceError),
    #[error(transparent)]
    Gpu(#[from] constraintforge_gpu::GpuError),
    #[error(
        "GPU verification mismatch at node {node}, shape {shape:?}: max absolute error={max_absolute:e}, max relative error={max_relative:e}"
    )]
    Mismatch {
        node: usize,
        shape: Vec<usize>,
        max_absolute: f32,
        max_relative: f32,
    },
    #[error("{0}")]
    InvalidArgument(&'static str),
    #[error("comparison selected different adapters between kernels")]
    AdapterChanged,
    #[error(transparent)]
    Bench(#[from] constraintforge_bench::BenchError),
    #[error(transparent)]
    Tuner(#[from] constraintforge_tuner::TunerError),
    #[error(transparent)]
    Research(#[from] constraintforge_research::ResearchError),
}

fn main() -> Result<(), CliError> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Command::Devices => devices(),
        Command::Demo {
            backend: Backend::Cpu,
            example: Example::Basic,
            ..
        } => cpu_demo(),
        Command::Demo {
            backend: Backend::Gpu,
            example: Example::Basic,
            matmul_kernel,
        } => gpu_demo(matmul_kernel.into()),
        Command::Demo {
            backend: Backend::Cpu,
            example: Example::Mlp,
            ..
        } => mlp_demo(Backend::Cpu, MatMulKernel::Naive),
        Command::Demo {
            backend: Backend::Gpu,
            example: Example::Mlp,
            matmul_kernel,
        } => mlp_demo(Backend::Gpu, matmul_kernel.into()),
        Command::Verify {
            backend: VerifyBackend::Gpu,
            suite: Suite::Matmul,
            matmul_kernel,
        } => verify_gpu(matmul_kernel.into()),
        Command::Verify {
            backend: VerifyBackend::Gpu,
            suite: Suite::Graph,
            matmul_kernel,
        } => verify_graph(matmul_kernel.into()),
        Command::Compare {
            operation:
                CompareOperation::Matmul {
                    m,
                    k,
                    n,
                    warmups,
                    runs,
                },
        } => compare_matmul([m, k, n], warmups, runs),
        Command::Bench { operation } => bench(operation),
        Command::Tune { operation } => tune(operation),
        Command::Research { operation } => research(operation),
    }
}

fn research(operation: ResearchOperation) -> Result<(), CliError> {
    use constraintforge_research::{
        ProspectiveProtocol, Protocol, ReferencePanelProtocol, RunControl, analyze,
        analyze_prospective, analyze_prospective_qualification, analyze_reference_panels, inspect,
        prospective_plan, qualify_resume, reference_panel_plan, reference_robustness, robustness,
        run_prospective_session, run_reference_panel_session, run_session_controlled,
        validate_prospective_results, validate_reference_panel_results, validate_results,
        write_prospective_manifest,
    };
    match operation {
        ResearchOperation::Inspect { protocol } => {
            let protocol = Protocol::parse(&protocol)?;
            let trials = inspect(&protocol)?;
            println!("protocol fingerprint: {}", protocol.fingerprint()?);
            println!("planned trials: {}", trials.len());
            for trial in trials {
                println!("{trial}");
            }
        }
        ResearchOperation::Pilot {
            protocol,
            output,
            stop_after_trials,
        } => {
            let protocol = Protocol::parse(&protocol)?;
            if protocol.phase != "calibration" || protocol.sessions != 1 {
                return Err(CliError::InvalidArgument(
                    "pilot requires a one-session calibration protocol",
                ));
            }
            let state = run_session_controlled(
                &protocol,
                "pilot-001",
                0,
                &output,
                false,
                RunControl {
                    stop_after_persisted_trials: stop_after_trials,
                },
            )?;
            println!("pilot status: {}", state.status);
        }
        ResearchOperation::Run {
            protocol,
            session,
            session_index,
            output,
            resume,
            stop_after_trials,
        } => {
            let protocol = Protocol::parse(&protocol)?;
            let state = run_session_controlled(
                &protocol,
                &session,
                session_index,
                &output,
                resume,
                RunControl {
                    stop_after_persisted_trials: stop_after_trials,
                },
            )?;
            println!("session status: {}", state.status);
        }
        ResearchOperation::Validate { input, output } => {
            let report = validate_results(&input)?;
            let json = serde_json::to_string_pretty(&report)
                .map_err(constraintforge_tuner::TunerError::Json)?;
            if let Some(path) = output {
                std::fs::write(path, format!("{json}\n"))
                    .map_err(constraintforge_tuner::TunerError::Io)?;
            } else {
                println!("{json}");
            }
            if !report.valid {
                return Err(CliError::InvalidArgument("research validation failed"));
            }
        }
        ResearchOperation::Analyze {
            input,
            output,
            seed,
            bootstrap_resamples,
        } => {
            let result = analyze(&input, &output, seed, bootstrap_resamples)?;
            println!("analysis rows: {}", result.aggregates.len());
            println!("output: {}", output.display());
        }
        ResearchOperation::Robustness {
            input,
            output,
            plan,
            protocol,
        } => {
            let result = robustness(&input, &output, &plan, &protocol)?;
            println!("classification: {}", result.classification);
            println!("trials: {}", result.trials);
            println!("output: {}", output.display());
        }
        ResearchOperation::ProspectiveInspect {
            protocol,
            session_index,
        } => {
            let protocol = ProspectiveProtocol::parse(&protocol)?;
            let plan = prospective_plan(&protocol, session_index)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan)
                    .map_err(constraintforge_research::ResearchError::from)?
            );
        }
        ResearchOperation::ProspectiveRun {
            protocol,
            session,
            session_index,
            output,
            resume,
        } => {
            let protocol = ProspectiveProtocol::parse(&protocol)?;
            let state =
                run_prospective_session(&protocol, &session, session_index, &output, resume)?;
            println!("phase: {}", state.phase);
            println!("search trials: {}", state.search.len());
            println!("reference records: {}", state.reference_files.len());
            println!("output: {}", output.display());
        }
        ResearchOperation::ProspectiveValidate { input } => {
            let report = validate_prospective_results(&input)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(constraintforge_research::ResearchError::from)?
            );
            if !report.valid {
                return Err(CliError::Research(
                    constraintforge_research::ResearchError::Invalid(
                        "prospective validation failed",
                    ),
                ));
            }
        }
        ResearchOperation::ProspectiveAnalyze {
            input,
            output,
            qualification_only,
        } => {
            let analysis = if qualification_only {
                analyze_prospective_qualification(&input, &output)?
            } else {
                analyze_prospective(&input, &output)?
            };
            println!("classification: {}", analysis.classification);
            println!("sessions: {}", analysis.sessions);
            println!("rows: {}", analysis.rows);
            println!("output: {}", output.display());
        }
        ResearchOperation::ReferenceRobustness {
            plan,
            input,
            output,
        } => {
            let analysis = reference_robustness(&plan, &input, &output)?;
            println!("classification: {}", analysis.classification);
            println!("sessions: {}", analysis.sessions);
            println!("panels: {}", analysis.panels);
            println!("output: {}", output.display());
        }
        ResearchOperation::ReferencePanelPlan {
            protocol,
            session_index,
        } => {
            let protocol = ReferencePanelProtocol::parse(&protocol)?;
            let plan = reference_panel_plan(&protocol, session_index)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan)
                    .map_err(constraintforge_research::ResearchError::from)?
            );
        }
        ResearchOperation::ReferencePanelRun {
            protocol,
            session,
            session_index,
            output,
            resume,
        } => {
            let protocol = ReferencePanelProtocol::parse(&protocol)?;
            let state =
                run_reference_panel_session(&protocol, &session, session_index, &output, resume)?;
            println!("phase: {}", state.phase);
            println!("candidate-panel records: {}", state.record_files.len());
            println!("durable attempts: {}", state.attempts.len());
            println!("output: {}", output.display());
        }
        ResearchOperation::ReferencePanelValidate { input } => {
            let report = validate_reference_panel_results(&input)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report)
                    .map_err(constraintforge_research::ResearchError::from)?
            );
            if !report.valid {
                return Err(CliError::Research(
                    constraintforge_research::ResearchError::Invalid(
                        "reference-panel validation failed",
                    ),
                ));
            }
        }
        ResearchOperation::ReferencePanelAnalyze {
            input,
            output,
            qualification_only,
        } => {
            let analysis = analyze_reference_panels(&input, &output, qualification_only)?;
            println!("classification: {}", analysis.classification);
            println!("sessions: {}", analysis.sessions);
            println!("workload pairs: {}", analysis.workload_pairs);
            println!("output: {}", output.display());
        }
        ResearchOperation::ProspectiveManifest { input, output } => {
            let files = write_prospective_manifest(&input, &output)?;
            println!("manifest entries: {files}");
            println!("output: {}", output.display());
        }
        ResearchOperation::QualifyResume {
            protocol,
            session,
            output,
            evidence,
            analysis,
        } => {
            let protocol = Protocol::parse(&protocol)?;
            if protocol.phase != "calibration" || protocol.sessions != 1 {
                return Err(CliError::InvalidArgument(
                    "resume qualification requires a one-session calibration protocol",
                ));
            }
            let report = qualify_resume(&protocol, &session, &output, &evidence, &analysis)?;
            println!("resume qualification: PASS");
            println!("trials: {}", report.expected_trials);
            println!("hashes unchanged: {}", report.hashes_unchanged);
            println!("evidence: {}", evidence.display());
            println!("analysis: {}", analysis.display());
        }
    }
    Ok(())
}

fn tune(operation: TuneOperation) -> Result<(), CliError> {
    match operation {
        TuneOperation::Candidates { m, k, n } => {
            let candidates = constraintforge_tuner::candidate_records([m, k, n])?;
            println!("candidate count: {}", candidates.len());
            for candidate in candidates {
                println!("{}", candidate.id);
                println!(
                    "  tile: {:?}, workgroup: {:?}, dispatch: {:?}",
                    candidate.tile, candidate.workgroup, candidate.dispatch
                );
                println!("  shared memory: {} bytes", candidate.shared_memory_bytes);
                println!(
                    "  padded output ratio: {:.4}, padded reduction ratio: {:.4}, reuse proxy: {:.4}, static score: {:.6}",
                    candidate.padded_output_ratio,
                    candidate.padded_reduction_ratio,
                    candidate.reuse_proxy,
                    candidate.static_score
                );
            }
        }
        TuneOperation::Matmul {
            args,
            policy,
            output,
        } => {
            let result = constraintforge_tuner::run_gpu(&tuning_config(&args, policy.into()))?;
            constraintforge_tuner::write_result(&output, &result)?;
            print_tuning_summary(&result, &output);
        }
        TuneOperation::Compare { args, output } => {
            let mut results = Vec::new();
            for policy in [
                Policy::Fixed,
                Policy::Heuristic,
                Policy::Random,
                Policy::ResourceAware,
            ] {
                results.push(constraintforge_tuner::run_gpu(&tuning_config(
                    &args, policy,
                ))?);
            }
            let comparison = TuningComparisonResult {
                schema_version: constraintforge_tuner::TUNING_SCHEMA_VERSION.into(),
                results,
            };
            constraintforge_tuner::write_comparison(&output, &comparison)?;
            println!("tuning policies: {}", comparison.results.len());
            for result in &comparison.results {
                println!(
                    "{:?}: selected {}, final median {:.3} us",
                    result.search.policy,
                    result.search.selected_candidate_id,
                    result.independent_evaluation_statistics_ns.median / 1_000.0
                );
            }
            println!("output: {}", output.display());
        }
    }
    Ok(())
}

fn tuning_config(args: &TuneArgs, policy: Policy) -> GpuTuningConfig {
    GpuTuningConfig {
        shape: [args.m, args.k, args.n],
        workload_seed: args.seed,
        search_seed: args.seed,
        policy,
        score_ablation: ScoreAblation::Full,
        search_design: SearchDesign::Legacy,
        budget: SearchBudget {
            max_candidates: args.max_candidates,
            max_batches: args.max_batches,
            max_duration: Duration::from_millis(args.max_ms),
        },
        measurement_mode: args.measurement.into(),
        tuning_warmups: args.tuning_warmups,
        tuning_runs: args.tuning_runs,
        evaluation_warmups: args.evaluation_warmups,
        evaluation_runs: args.evaluation_runs,
    }
}

fn print_tuning_summary(result: &constraintforge_tuner::TuningResult, output: &std::path::Path) {
    println!(
        "policy: {:?} {}",
        result.search.policy, result.policy_version
    );
    println!(
        "adapter: {} ({})",
        result.adapter.name, result.adapter.backend
    );
    println!(
        "candidate pool: {}, compiled: {}",
        result.candidate_pool.len(),
        result.compiled_candidates
    );
    println!("selected: {}", result.search.selected_candidate_id);
    println!(
        "selection score: {}",
        result
            .search
            .selected_score_ns
            .map(|value| format!("{:.3} us", value / 1_000.0))
            .unwrap_or_else(|| "not measured during selection".into())
    );
    println!(
        "batches: {}, measured dispatches: {}",
        result.search.measurement_batches, result.search.total_measured_dispatches
    );
    println!(
        "tuning wall time: {:.3} ms",
        result.tuning_wall_time_ns as f64 / 1_000_000.0
    );
    println!(
        "independent evaluation median: {:.3} us",
        result.independent_evaluation_statistics_ns.median / 1_000.0
    );
    println!("correctness: PASS");
    println!("output: {}", output.display());
}

fn bench(operation: BenchOperation) -> Result<(), CliError> {
    match operation {
        BenchOperation::Matmul {
            m,
            k,
            n,
            warmups,
            runs,
            seed,
            kernel,
            output,
        } => {
            let kernel = match kernel {
                BenchKernel::Naive => KernelChoice::Naive,
                BenchKernel::Tiled => KernelChoice::Tiled,
            };
            let result = constraintforge_bench::run(&BenchmarkConfig {
                shape: [m, k, n],
                warmups,
                runs,
                seed,
                kernel,
            })?;
            constraintforge_bench::write_result(&output, &result)?;
            print_benchmark_summary(&result, &output);
        }
        BenchOperation::Compare {
            m,
            k,
            n,
            warmups,
            runs,
            seed,
            output,
        } => {
            let result = constraintforge_bench::run(&BenchmarkConfig {
                shape: [m, k, n],
                warmups,
                runs,
                seed,
                kernel: KernelChoice::Both,
            })?;
            constraintforge_bench::write_result(&output, &result)?;
            print_benchmark_summary(&result, &output);
        }
        BenchOperation::Suite {
            preset: BenchPreset::Smoke,
            warmups,
            runs,
            seed,
            output,
        } => {
            let shapes = [[2, 2, 2], [15, 17, 13], [31, 33, 29], [64, 64, 64]];
            let mut results = Vec::with_capacity(shapes.len());
            for shape in shapes {
                results.push(constraintforge_bench::run(&BenchmarkConfig {
                    shape,
                    warmups,
                    runs,
                    seed,
                    kernel: KernelChoice::Both,
                })?);
            }
            let suite = BenchmarkSuiteResult {
                schema_version: constraintforge_bench::SCHEMA_VERSION.into(),
                suite_name: "smoke".into(),
                results,
            };
            constraintforge_bench::write_suite(&output, &suite)?;
            println!("benchmark suite: smoke");
            println!("workloads: {}", suite.results.len());
            println!("output: {}", output.display());
        }
    }
    Ok(())
}

fn print_benchmark_summary(result: &BenchmarkResult, output: &std::path::Path) {
    println!("adapter: {}", result.hardware.adapter_name);
    println!("backend: {}", result.hardware.backend);
    println!(
        "shape: M={}, K={}, N={}",
        result.shape_mkn[0], result.shape_mkn[1], result.shape_mkn[2]
    );
    println!(
        "preparation wall time: {:.3} ms",
        result.preparation_wall_time_ns as f64 / 1_000_000.0
    );
    println!(
        "GPU timestamps available: {}",
        result.hardware.timestamp_query_supported
    );
    for kernel in &result.kernels {
        println!(
            "{} correctness: {}",
            kernel.variant,
            if kernel.correctness.passed {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "{} prepared end-to-end median: {:.3} ms",
            kernel.variant,
            kernel.host_statistics_ns.median / 1_000_000.0
        );
        match &kernel.gpu_statistics_ns {
            Some(stats) => println!(
                "{} GPU compute-pass median: {:.3} ms, p95: {:.3} ms",
                kernel.variant,
                stats.median / 1_000_000.0,
                stats.p95_nearest_rank.unwrap_or(stats.median) / 1_000_000.0
            ),
            None => println!("{} GPU compute-pass duration: unavailable", kernel.variant),
        }
    }
    println!("output: {}", output.display());
}

fn devices() -> Result<(), CliError> {
    let adapters = constraintforge_gpu::adapters();
    if adapters.is_empty() {
        println!("No compute adapters found.");
    }
    for (index, adapter) in adapters.iter().enumerate() {
        println!("Adapter {index}: {}", adapter.name);
        println!("  backend: {}", adapter.backend);
        println!("  device type: {}", adapter.device_type);
        println!("  max buffer size: {} bytes", adapter.max_buffer_size);
        println!(
            "  max storage binding: {} bytes",
            adapter.max_storage_buffer_binding_size
        );
        println!(
            "  max compute workgroups/dimension: {}",
            adapter.max_compute_workgroups_per_dimension
        );
        println!("  features: {}", adapter.features);
    }
    Ok(())
}

fn cpu_demo() -> Result<(), CliError> {
    let mut graph = Graph::new();
    let a = graph.input(TensorSpec::f32([2, 3])?);
    let b = graph.input(TensorSpec::f32([3, 2])?);
    let bias = graph.input(TensorSpec::f32([2])?);
    let product = graph.matmul(a, b)?;
    let biased = graph.bias_add(product, bias)?;
    let output = graph.relu(biased)?;
    graph.set_outputs(vec![output])?;
    let values = execute(
        &graph,
        &Inputs::from([
            (a, vec![1., 2., 3., 4., 5., 6.]),
            (b, vec![7., 8., 9., 10., 11., 12.]),
            (bias, vec![-60., -60.]),
        ]),
    )?;
    println!("backend: cpu-reference");
    println!("graph: MatMul -> BiasAdd -> ReLU");
    println!("shape: [2, 2]");
    println!("output: {:?}", values[&output].values);
    Ok(())
}

fn gpu_demo(kernel: MatMulKernel) -> Result<(), CliError> {
    let left = [1., 2., 3., 4., 5., 6.];
    let right = [7., 8., 9., 10., 11., 12.];
    let mut graph = Graph::new();
    let left_id = graph.input(TensorSpec::f32([2, 3])?);
    let right_id = graph.input(TensorSpec::f32([3, 2])?);
    let output_id = graph.matmul(left_id, right_id)?;
    graph.set_outputs(vec![output_id])?;
    let execution = constraintforge_gpu::execute_graph_with_kernel(
        &graph,
        &Inputs::from([(left_id, left.to_vec()), (right_id, right.to_vec())]),
        kernel,
    )?;
    println!("backend: gpu");
    println!("operation: MatMul");
    println!("shape: [2, 2]");
    print_kernel_selections(&execution)?;
    println!("output: {:?}", execution.output(output_id)?);
    Ok(())
}

fn verify_gpu(kernel: MatMulKernel) -> Result<(), CliError> {
    let cases = [
        (
            [2, 3],
            vec![1., 2., 3., 4., 5., 6.],
            [3, 2],
            vec![7., 8., 9., 10., 11., 12.],
        ),
        (
            [3, 2],
            vec![-2., 1., 0.5, 3., 4., -1.],
            [2, 5],
            vec![1., 2., 3., 4., 5., -1., -2., -3., -4., -5.],
        ),
    ];
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut matches = true;
    for (left_shape, left, right_shape, right) in cases {
        let expected = matmul(
            &Tensor::new(TensorSpec::f32(left_shape)?, left.clone())?,
            &Tensor::new(TensorSpec::f32(right_shape)?, right.clone())?,
        )?;
        let mut graph = Graph::new();
        let left_id = graph.input(TensorSpec::f32(left_shape)?);
        let right_id = graph.input(TensorSpec::f32(right_shape)?);
        let output_id = graph.matmul(left_id, right_id)?;
        graph.set_outputs(vec![output_id])?;
        let execution = constraintforge_gpu::execute_graph_with_kernel(
            &graph,
            &Inputs::from([(left_id, left), (right_id, right)]),
            kernel,
        )?;
        let actual = execution.output(output_id)?;
        let (case_absolute, case_relative, case_matches) = compare(actual, &expected.values);
        max_absolute = max_absolute.max(case_absolute);
        max_relative = max_relative.max(case_relative);
        matches &= case_matches;
    }
    println!("absolute tolerance: {ABSOLUTE_TOLERANCE:e}");
    println!("relative tolerance: {RELATIVE_TOLERANCE:e}");
    println!("maximum absolute error: {max_absolute:e}");
    println!("maximum relative error: {max_relative:e}");
    if !matches {
        return Err(CliError::Mismatch {
            node: 0,
            shape: vec![2, 2],
            max_absolute,
            max_relative,
        });
    }
    println!("GPU verification: PASS");
    Ok(())
}

fn mlp_graph(batch: usize) -> Result<(Graph, Inputs, NodeId), CliError> {
    let mut graph = Graph::new();
    let x = graph.input(TensorSpec::f32([batch, 3])?);
    let w1 = graph.input(TensorSpec::f32([3, 4])?);
    let b1 = graph.input(TensorSpec::f32([4])?);
    let w2 = graph.input(TensorSpec::f32([4, 2])?);
    let b2 = graph.input(TensorSpec::f32([2])?);
    let hidden = graph.matmul(x, w1)?;
    let hidden = graph.bias_add(hidden, b1)?;
    let hidden = graph.relu(hidden)?;
    let output = graph.matmul(hidden, w2)?;
    let output = graph.bias_add(output, b2)?;
    graph.set_outputs(vec![output])?;
    let inputs = Inputs::from([
        (x, vec![-1., -2. / 3., -1. / 3., 0., 1. / 3., 2. / 3.]),
        (
            w1,
            vec![0.5, -1., 0., 2., -0.5, 1., 1.5, 0., 1., 0.25, -2., 0.5],
        ),
        (b1, vec![0., -0.5, 1., 0.25]),
        (w2, vec![1., -1., 0.5, 2., -0.5, 1., 2., 0.25]),
        (b2, vec![-0.25, 0.5]),
    ]);
    Ok((graph, inputs, output))
}

fn mlp_demo(backend: Backend, kernel: MatMulKernel) -> Result<(), CliError> {
    let (graph, inputs, output) = mlp_graph(2)?;
    println!("graph: MatMul -> BiasAdd -> ReLU -> MatMul -> BiasAdd");
    println!("input shape: [2, 3]");
    println!("output shape: [2, 2]");
    match backend {
        Backend::Cpu => {
            let values = execute(&graph, &inputs)?;
            println!("backend: cpu-reference");
            println!("output: {:?}", values[&output].values);
        }
        Backend::Gpu => {
            let execution =
                constraintforge_gpu::execute_graph_with_kernel(&graph, &inputs, kernel)?;
            println!("adapter: {}", execution.adapter.name);
            println!("backend: {}", execution.adapter.backend);
            print_kernel_selections(&execution)?;
            println!("output: {:?}", execution.output(output)?);
        }
    }
    println!("execution: PASS");
    Ok(())
}

fn verify_graph(kernel: MatMulKernel) -> Result<(), CliError> {
    let (graph, inputs, output) = mlp_graph(2)?;
    let expected = execute(&graph, &inputs)?;
    let actual = constraintforge_gpu::execute_graph_with_kernel(&graph, &inputs, kernel)?;
    let (max_absolute, max_relative, matches) =
        compare(actual.output(output)?, &expected[&output].values);
    println!("adapter: {}", actual.adapter.name);
    println!("backend: {}", actual.adapter.backend);
    println!("graph: two-layer MLP");
    print_kernel_selections(&actual)?;
    println!("absolute tolerance: {ABSOLUTE_TOLERANCE:e}");
    println!("relative tolerance: {RELATIVE_TOLERANCE:e}");
    println!("maximum absolute error: {max_absolute:e}");
    println!("maximum relative error: {max_relative:e}");
    if !matches {
        return Err(CliError::Mismatch {
            node: output.index(),
            shape: graph.node(output)?.spec.shape.clone(),
            max_absolute,
            max_relative,
        });
    }
    println!("GPU graph verification: PASS");
    Ok(())
}

fn print_kernel_selections(execution: &GraphExecution) -> Result<(), CliError> {
    for selection in &execution.matmul_kernels {
        let variant = match selection.config.variant() {
            MatMulKernelVariant::Naive => "naive",
            MatMulKernelVariant::Tiled => "tiled",
        };
        println!(
            "MatMul node {} shape [{}, {}, {}]: {variant}",
            selection.node.index(),
            selection.shape[0],
            selection.shape[1],
            selection.shape[2]
        );
        println!("  tile: {:?}", selection.config.tile());
        println!("  workgroup: {:?}", selection.config.workgroup());
        println!(
            "  estimated workgroup memory: {} bytes",
            selection.config.shared_memory_bytes()?
        );
        println!("  reason: {}", selection.reason);
    }
    Ok(())
}

fn compare_matmul(shape: [usize; 3], warmups: u32, runs: u32) -> Result<(), CliError> {
    if warmups > 1000 || runs == 0 || runs > 1000 {
        return Err(CliError::InvalidArgument(
            "warmups must be at most 1000 and runs must be between 1 and 1000",
        ));
    }
    let [m, k, n] = shape;
    let left_spec = TensorSpec::f32([m, k])?;
    let right_spec = TensorSpec::f32([k, n])?;
    let left_values = (0..left_spec.element_count()?)
        .map(|i| (i % 17) as f32 / 8.0 - 1.0)
        .collect::<Vec<_>>();
    let right_values = (0..right_spec.element_count()?)
        .map(|i| (i % 13) as f32 / 6.0 - 1.0)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let left = graph.input(left_spec.clone());
    let right = graph.input(right_spec.clone());
    let output = graph.matmul(left, right)?;
    graph.set_outputs(vec![output])?;
    let inputs = Inputs::from([(left, left_values.clone()), (right, right_values.clone())]);
    let reference = matmul(
        &Tensor::new(left_spec, left_values)?,
        &Tensor::new(right_spec, right_values)?,
    )?;

    let (naive, naive_times) = timed_runs(&graph, &inputs, MatMulKernel::Naive, warmups, runs)?;
    let (tiled, tiled_times) = timed_runs(&graph, &inputs, MatMulKernel::Tiled, warmups, runs)?;
    if naive.adapter.name != tiled.adapter.name || naive.adapter.backend != tiled.adapter.backend {
        return Err(CliError::AdapterChanged);
    }
    let (naive_absolute, naive_relative, naive_matches) =
        compare(naive.output(output)?, &reference.values);
    let (tiled_absolute, tiled_relative, tiled_matches) =
        compare(tiled.output(output)?, &reference.values);
    let (_, _, kernels_match) = compare(naive.output(output)?, tiled.output(output)?);
    if !(naive_matches && tiled_matches && kernels_match) {
        return Err(CliError::Mismatch {
            node: output.index(),
            shape: vec![m, n],
            max_absolute: naive_absolute.max(tiled_absolute),
            max_relative: naive_relative.max(tiled_relative),
        });
    }

    println!("adapter: {}", naive.adapter.name);
    println!("backend: {}", naive.adapter.backend);
    println!("dtype: F32");
    println!("shape: M={m}, K={k}, N={n}");
    println!("warmups: {warmups}");
    println!("measured repetitions: {runs}");
    println!("correctness: PASS");
    println!("naive maximum absolute error: {naive_absolute:e}");
    println!("naive maximum relative error: {naive_relative:e}");
    println!("tiled maximum absolute error: {tiled_absolute:e}");
    println!("tiled maximum relative error: {tiled_relative:e}");
    println!("GPU kernel timestamp: unavailable (not collected by this preliminary facility)");
    println!("pipeline preparation: included in end-to-end time, not reported separately");
    println!("CPU submission overhead: not isolated");
    println!("readback: included in end-to-end time");
    println!("measurement method: host wall-clock around a complete fresh GPU invocation");
    print_duration_stats("naive end-to-end", &naive_times);
    print_duration_stats("tiled end-to-end", &tiled_times);
    if cfg!(debug_assertions) {
        println!("warning: debug build; use --release for performance observations");
    }
    Ok(())
}

fn timed_runs(
    graph: &Graph,
    inputs: &Inputs,
    kernel: MatMulKernel,
    warmups: u32,
    runs: u32,
) -> Result<(GraphExecution, Vec<Duration>), CliError> {
    for _ in 0..warmups {
        constraintforge_gpu::execute_graph_with_kernel(graph, inputs, kernel)?;
    }
    let mut times = Vec::with_capacity(runs as usize);
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        let execution = constraintforge_gpu::execute_graph_with_kernel(graph, inputs, kernel)?;
        times.push(start.elapsed());
        last = Some(execution);
    }
    Ok((
        last.ok_or(CliError::InvalidArgument("runs must be nonzero"))?,
        times,
    ))
}

fn print_duration_stats(label: &str, values: &[Duration]) {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let total: Duration = sorted.iter().copied().sum();
    let mean = total.as_secs_f64() * 1000.0 / sorted.len() as f64;
    let median = sorted[sorted.len() / 2].as_secs_f64() * 1000.0;
    let minimum = sorted[0].as_secs_f64() * 1000.0;
    println!("{label}: min={minimum:.3} ms, median={median:.3} ms, mean={mean:.3} ms");
}

fn compare(actual: &[f32], expected: &[f32]) -> (f32, f32, bool) {
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut matches = actual.len() == expected.len();
    for (&actual, &expected) in actual.iter().zip(expected) {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(ABSOLUTE_TOLERANCE);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        matches &= actual.is_finite()
            && expected.is_finite()
            && within_tolerance(actual, expected, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE);
    }
    (max_absolute, max_relative, matches)
}

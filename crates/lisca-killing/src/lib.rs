//! Killing analysis for LiSCA workspaces (`assay.json` + `roi/`).
//!
//! This crate owns death-reporter fluorescence, PI-step viability and survival
//! curves, fluorescent engagement, the killing classifier (predict, shard
//! merge, clean), and killing assay figures. The lisca monorepo depends on it
//! via git URL. This crate must not depend on the `lisca` crate.
//!
//! Folder names match lisca (`roi/Pos{n}/`, `analysis/Pos{n}/`, `results/<sample>/`).
//! Crop stays in lisca. Predict loads `model.onnx` only with the `onnx` feature.
//!
//! Callers pass a workspace path and a [`SampleMapping`]. ndarray types are
//! not part of this API.

mod array;
mod clean;
mod csv_io;
mod engagement;
mod export;
mod fluorescence;
mod plot;
mod predict;
mod roi;
mod sample;
mod shards;
mod traces_pack;
mod viability;

use font_kit as _;
use std::path::Path;

pub use array::{full_frame_roi_stats, MaskedRoiStats};
pub use clean::{run_clean, CLEAN_THRESHOLD};
pub use engagement::{
    component_count, count_engagements, discover_engagement_csvs, label_components,
    mask_with_scale, run_position_engagement, split_round_components, tcell_mask, tumor_mask,
    write_engagement_summary, EngagementCounts, FluorescenceHistogram, FluorescenceScale,
    ENGAGER_DIAMETER_PX,
};
pub use fluorescence::{run_fluorescence, run_position_fluorescence};
pub use plot::{
    run_plot_death_times, run_plot_engagement_traces, run_plot_kill, run_plot_survival,
    run_plot_traces,
};
pub use predict::{
    alive_label, dead_probability, run_predict_to, run_predict_to_controlled, PredictControl,
    PredictFailure, PredictOptions, DEAD_LABEL_THRESHOLD,
};
pub use sample::{SampleAnalysis, SampleMapping};
pub use shards::merge_prediction_shards;
pub use traces_pack::{parse_trace_path, publish_sample_traces_xlsx, resolve_sample};
pub use viability::run_viability;

/// Synchronous full `killing-death-reporter` pipeline (fluorescence -> traces
/// plots -> PI-step viability -> survival, kill, and death-time figures).
pub fn run_death_reporter_pipeline(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
) -> Result<(), String> {
    run_fluorescence(workspace, mapping)?;
    run_plot_traces(workspace, mapping, interval, None)?;
    run_viability(workspace, mapping, interval)?;
    run_plot_survival(workspace, mapping, interval)?;
    run_plot_kill(workspace, mapping, interval)?;
    run_plot_death_times(workspace, mapping, interval)
}

/// Synchronous `killing-engagement` measurement and workbook summary pipeline.
pub fn run_engagement_pipeline(
    workspace: &Path,
    mapping: &SampleMapping,
    interval_minutes: f64,
) -> Result<(), String> {
    for position in mapping.positions() {
        run_position_engagement(workspace, mapping, position, interval_minutes)?;
    }
    write_engagement_summary(workspace, mapping)
}

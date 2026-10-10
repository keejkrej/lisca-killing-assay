//! Killing analysis for LiSCA workspaces (`assay.json` + `roi/`).
//!
//! This crate is the Rust port of the Python `killing` package: death-reporter
//! fluorescence, fluorescent engagement, and the killing classifier (predict,
//! clean, kill curve). The lisca monorepo depends on it via git URL. This crate
//! must not depend on the `lisca` crate.
//!
//! Folder names match lisca (`roi/Pos{n}/`, `analysis/Pos{n}/`, `results/<sample>/`).
//! Crop stays in lisca. Studio schedules the steps and draws the figures.
//! Predict loads `model.onnx` only with the `onnx` feature.
//!
//! Callers pass a workspace path and a [`SampleMapping`]. ndarray types are
//! not part of this API.

mod array;
mod clean;
mod csv_io;
mod engagement;
mod export;
mod fluorescence;
mod predict;
mod roi;
mod sample;

pub use array::{full_frame_roi_stats, MaskedRoiStats};
pub use clean::{run_clean, CLEAN_THRESHOLD};
pub use engagement::{
    component_count, count_engagements, discover_engagement_csvs, label_components,
    mask_with_scale, run_position_engagement, split_round_components, tcell_mask, tumor_mask,
    write_engagement_summary, EngagementCounts, FluorescenceHistogram, FluorescenceScale,
    ENGAGER_DIAMETER_PX,
};
pub use fluorescence::{run_fluorescence, run_position_fluorescence};
pub use predict::{
    alive_label, dead_probability, run_predict_to, run_predict_to_controlled, PredictControl,
    PredictFailure, PredictOptions, DEAD_LABEL_THRESHOLD,
};
pub use sample::{SampleAnalysis, SampleMapping};

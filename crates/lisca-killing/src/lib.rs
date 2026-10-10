//! Killing analysis for LiSCA workspaces (`assay.json` + `roi/`).
//!
//! This crate is the Rust port of the death-reporter and fluorescent-engagement
//! code in the Python `apoptosis` package. The lisca monorepo depends on it
//! via git URL. This crate must not depend on the `lisca` crate.
//!
//! Folder names match lisca (`roi/Pos{n}/`, `analysis/Pos{n}/`, `results/<sample>/`).
//! Crop stays in lisca. Studio schedules the steps and draws the shared
//! per-sample figures; the measurement and the engagement counts live here.
//!
//! Callers pass a workspace path and a [`SampleMapping`]. ndarray types are
//! not part of this API.

mod array;
mod csv_io;
mod engagement;
mod export;
mod fluorescence;
mod roi;
mod sample;

pub use array::{full_frame_roi_stats, MaskedRoiStats};
pub use engagement::{
    component_count, count_engagements, discover_engagement_csvs, label_components,
    mask_with_scale, run_position_engagement, split_round_components, tcell_mask, tumor_mask,
    write_engagement_summary, EngagementCounts, FluorescenceHistogram, FluorescenceScale,
    ENGAGER_DIAMETER_PX,
};
pub use fluorescence::{run_fluorescence, run_position_fluorescence};
pub use sample::{SampleAnalysis, SampleMapping};

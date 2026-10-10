//! Killing classifier: alive/dead label and the ResNet predict loop.
//!
//! `label` true means a cell is still present. `p_dead` is the model's
//! probability of the absent class. A crop is alive while `p_dead` is below
//! [`DEAD_LABEL_THRESHOLD`]. Loading `model.onnx` and running the session
//! needs the `onnx` feature. Clean, death time, and the kill curve are
//! [`crate::clean`].

use std::path::Path;

use crate::sample::SampleMapping;

pub const DEAD_LABEL_THRESHOLD: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct PredictOptions {
    pub batch_size: usize,
}

impl Default for PredictOptions {
    fn default() -> Self {
        // Batch 256 makes the ResNet18 activation tensors hundreds of megabytes.
        // 32 matches transfection segment and stays small on an 8 GB laptop.
        Self { batch_size: 32 }
    }
}

/// Stop and progress hooks for a predict shard. The task scheduler cannot abort
/// `spawn_blocking`, so the batch loop has to notice cancellation itself.
pub struct PredictControl<'a> {
    pub is_cancelled: &'a dyn Fn() -> bool,
    pub on_frames: &'a dyn Fn(u32, u32),
}

#[derive(Debug, PartialEq, Eq)]
pub enum PredictFailure {
    Cancelled,
    Failed(String),
}

impl From<String> for PredictFailure {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

impl From<&str> for PredictFailure {
    fn from(message: &str) -> Self {
        Self::Failed(message.to_string())
    }
}

/// P(dead) = P(absent). The first logit is the absent class.
pub fn dead_probability(absent_logit: f32, present_logit: f32) -> f64 {
    let max = absent_logit.max(present_logit);
    let first_exp = (absent_logit - max).exp();
    let second_exp = (present_logit - max).exp();
    f64::from(first_exp / (first_exp + second_exp))
}

/// `label` true means a present (alive) cell. Dead once `p_dead` meets the threshold.
pub fn alive_label(p_dead: f64) -> bool {
    p_dead < DEAD_LABEL_THRESHOLD
}

#[cfg(any(test, feature = "onnx"))]
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) struct FrameBatchItem {
    pub(crate) pos: u32,
    pub(crate) sample: String,
    pub(crate) signal_channel: u32,
    pub(crate) roi: u32,
    pub(crate) t: u32,
    pub(crate) pixels: Vec<f64>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Decoded frames waiting for inference. Callers flush once `limit` frames are buffered.
#[cfg(any(test, feature = "onnx"))]
pub(crate) struct FrameBatch {
    frames: Vec<FrameBatchItem>,
    limit: usize,
}

#[cfg(any(test, feature = "onnx"))]
impl FrameBatch {
    pub(crate) fn new(limit: usize) -> Self {
        let limit = limit.max(1);
        Self {
            frames: Vec::with_capacity(limit),
            limit,
        }
    }

    pub(crate) fn push(&mut self, frame: FrameBatchItem) -> Option<Vec<FrameBatchItem>> {
        self.frames.push(frame);
        if self.frames.len() >= self.limit {
            Some(std::mem::take(&mut self.frames))
        } else {
            None
        }
    }

    pub(crate) fn drain_remainder(&mut self) -> Option<Vec<FrameBatchItem>> {
        if self.frames.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.frames))
        }
    }
}

/// Run inference in batches. `completed_offset` / `total` are the shard-wide
/// frame counts so a second channel does not reset the progress bar.
#[cfg(any(test, feature = "onnx"))]
pub(crate) fn run_frame_batches(
    frames: &[FrameBatchItem],
    batch_size: usize,
    completed_offset: u32,
    total: u32,
    is_cancelled: &dyn Fn() -> bool,
    on_frames: &dyn Fn(u32, u32),
    mut infer_batch: impl FnMut(&[FrameBatchItem]) -> Result<Vec<f64>, String>,
) -> Result<Vec<f64>, PredictFailure> {
    let mut probabilities = Vec::with_capacity(frames.len());
    let mut completed = completed_offset;
    for chunk in frames.chunks(batch_size.max(1)) {
        if is_cancelled() {
            return Err(PredictFailure::Cancelled);
        }
        let batch = infer_batch(chunk)?;
        if batch.len() != chunk.len() {
            return Err(PredictFailure::Failed(
                "kill model returned a different number of predictions than frames".to_string(),
            ));
        }
        let chunk_len = u32::try_from(chunk.len()).unwrap_or(u32::MAX);
        completed = completed.saturating_add(chunk_len).min(total);
        probabilities.extend(batch);
        on_frames(completed, total);
    }
    if is_cancelled() {
        return Err(PredictFailure::Cancelled);
    }
    Ok(probabilities)
}

#[cfg(feature = "onnx")]
mod onnx;

pub fn run_predict_to(
    workspace: &Path,
    output_workspace: &Path,
    mapping: &SampleMapping,
    model_dir: &Path,
    options: PredictOptions,
) -> Result<(), String> {
    run_predict_to_controlled(
        workspace,
        output_workspace,
        mapping,
        model_dir,
        options,
        &PredictControl {
            is_cancelled: &|| false,
            on_frames: &|_, _| {},
        },
    )
    .map_err(|error| match error {
        PredictFailure::Cancelled => "prediction cancelled".to_string(),
        PredictFailure::Failed(message) => message,
    })
}

pub fn run_predict_to_controlled(
    workspace: &Path,
    output_workspace: &Path,
    mapping: &SampleMapping,
    model_dir: &Path,
    options: PredictOptions,
    control: &PredictControl<'_>,
) -> Result<(), PredictFailure> {
    if (control.is_cancelled)() {
        return Err(PredictFailure::Cancelled);
    }
    #[cfg(feature = "onnx")]
    {
        onnx::run(
            workspace,
            output_workspace,
            mapping,
            model_dir,
            options,
            control,
        )
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (workspace, output_workspace, mapping, model_dir, options);
        Err(PredictFailure::Failed(
            "killing predict requires the `onnx` feature".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_probability_prefers_absent_label() {
        let probability = dead_probability(2.0, 0.0);
        assert!(probability > 0.8);
    }

    #[test]
    fn dead_probability_prefers_present_label() {
        let probability = dead_probability(0.0, 2.0);
        assert!(probability < 0.2);
    }

    #[test]
    fn alive_label_true_when_dead_probability_below_threshold() {
        assert!(alive_label(0.1));
        assert!(alive_label(0.49));
        assert!(alive_label(0.0));
    }

    #[test]
    fn alive_label_false_when_dead_probability_meets_threshold() {
        assert!(!alive_label(0.5));
        assert!(!alive_label(0.9));
        assert!(!alive_label(1.0));
    }

    #[test]
    fn controlled_predict_stops_before_loading_a_model() {
        let frames_reported = std::cell::Cell::new(false);
        let error = run_predict_to_controlled(
            Path::new("/missing"),
            Path::new("/missing"),
            &SampleMapping::default(),
            Path::new("/missing"),
            PredictOptions::default(),
            &PredictControl {
                is_cancelled: &|| true,
                on_frames: &|_, _| frames_reported.set(true),
            },
        )
        .expect_err("cancel before work");
        assert_eq!(error, PredictFailure::Cancelled);
        assert!(!frames_reported.get());
    }

    #[test]
    fn frame_batches_stop_before_the_next_batch_and_report_progress() {
        let frames = vec![
            frame_item(0),
            frame_item(1),
            frame_item(2),
            frame_item(3),
            frame_item(4),
        ];
        let batches = std::cell::Cell::new(0);
        let seen = std::cell::RefCell::new(Vec::new());
        let error = run_frame_batches(
            &frames,
            2,
            0,
            5,
            &|| batches.get() >= 1,
            &|completed, total| seen.borrow_mut().push((completed, total)),
            |chunk| {
                batches.set(batches.get() + 1);
                Ok(vec![0.25; chunk.len()])
            },
        )
        .expect_err("stop after the first batch");
        assert_eq!(error, PredictFailure::Cancelled);
        assert_eq!(batches.get(), 1);
        assert_eq!(seen.into_inner(), vec![(2, 5)]);
    }

    fn frame_item(t: u32) -> FrameBatchItem {
        FrameBatchItem {
            pos: 1,
            sample: "sample".to_string(),
            signal_channel: 0,
            roi: 1,
            t,
            pixels: vec![0.0],
            width: 1,
            height: 1,
        }
    }

    #[test]
    fn frame_batch_flushes_at_the_limit_instead_of_retaining_every_frame() {
        let mut pending = FrameBatch::new(2);
        let mut flushed = Vec::new();
        for t in 0..5 {
            if let Some(full) = pending.push(frame_item(t)) {
                assert!(
                    full.len() <= 2,
                    "a flush must not grow past the batch limit"
                );
                flushed.push(full.len());
            }
            assert!(
                pending.frames.len() < 2,
                "unflushed frames must stay under the batch limit"
            );
        }
        let tail = pending.drain_remainder().expect("partial tail");
        assert_eq!(flushed, vec![2, 2]);
        assert_eq!(tail.len(), 1);
        assert!(pending.drain_remainder().is_none());
    }
}

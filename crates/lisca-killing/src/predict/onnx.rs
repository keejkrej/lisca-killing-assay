//! ONNX ResNet18 predict. Compiled only with the `onnx` feature.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use image::imageops::FilterType;
use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array, ArrayView, Axis, Ix4};
use ort::session::Session;
use ort::value::Tensor;

use crate::csv_io::{format_float, write_csv_only};
use crate::roi::{
    position_dir, read_position_index, roi_frame_2d, validate_channel_index, RoiStack,
};
use crate::sample::SampleMapping;

use super::{
    alive_label, run_frame_batches, FrameBatch, FrameBatchItem, PredictControl, PredictFailure,
    PredictOptions,
};

const IMAGE_SIZE: u32 = 224;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn run(
    workspace: &Path,
    output_workspace: &Path,
    mapping: &SampleMapping,
    model_dir: &Path,
    options: PredictOptions,
    control: &PredictControl<'_>,
) -> Result<(), PredictFailure> {
    let mut total_frames = 0u32;
    for sample in mapping {
        for &signal_channel in &sample.signal {
            for position in &sample.positions {
                if (control.is_cancelled)() {
                    return Err(PredictFailure::Cancelled);
                }
                total_frames = total_frames.saturating_add(count_position_frames(
                    workspace,
                    signal_channel,
                    *position,
                )?);
            }
        }
    }
    (control.on_frames)(0, total_frames);
    if (control.is_cancelled)() {
        return Err(PredictFailure::Cancelled);
    }

    let model_path = model_dir.join("model.onnx");
    if !model_path.is_file() {
        return Err(PredictFailure::Failed(format!(
            "missing kill model at {}",
            model_path.display()
        )));
    }

    let mut session = build_kill_session(&model_path)?;
    let input_name = session
        .inputs()
        .first()
        .ok_or("kill model has no inputs")?
        .name()
        .to_string();

    let mut traces_by_pos_channel: BTreeMap<(u32, u32), Vec<TraceRow>> = BTreeMap::new();
    let mut prediction_rows: Vec<PredictionRow> = Vec::new();
    let mut completed_frames = 0u32;
    let mut pending = FrameBatch::new(options.batch_size);
    let mut commit_batch = |frames: Vec<FrameBatchItem>| -> Result<(), PredictFailure> {
        if frames.is_empty() {
            return Ok(());
        }
        let probabilities = run_frame_batches(
            &frames,
            frames.len().max(1),
            completed_frames,
            total_frames,
            control.is_cancelled,
            control.on_frames,
            |chunk| run_batch_inference(&mut session, &input_name, chunk),
        )?;
        let frame_count = u32::try_from(frames.len()).unwrap_or(u32::MAX);
        completed_frames = completed_frames
            .saturating_add(frame_count)
            .min(total_frames);
        for (frame, p_dead) in frames.into_iter().zip(probabilities) {
            traces_by_pos_channel
                .entry((frame.pos, frame.signal_channel))
                .or_default()
                .push(TraceRow {
                    pos: frame.pos,
                    roi: frame.roi,
                    t: frame.t,
                    p_dead,
                });
            prediction_rows.push(PredictionRow {
                t: frame.t,
                roi: frame.roi,
                p_dead,
                label: alive_label(p_dead),
                pos: frame.pos,
                sample: frame.sample,
            });
        }
        Ok(())
    };

    for sample in mapping {
        for &signal_channel in &sample.signal {
            for position in &sample.positions {
                if (control.is_cancelled)() {
                    return Err(PredictFailure::Cancelled);
                }
                let pos_dir = match position_dir(workspace, *position) {
                    Ok(path) => path,
                    Err(_) => continue,
                };
                let index = read_position_index(&pos_dir)?;
                validate_channel_index(&index, signal_channel)?;
                for roi_crop in &index.rois {
                    if (control.is_cancelled)() {
                        return Err(PredictFailure::Cancelled);
                    }
                    // One ROI stack at a time. It is dropped before the next
                    // file is decoded, and frames are inferred every batch
                    // instead of retained for the whole position.
                    let stack = RoiStack::load(&pos_dir.join(&roi_crop.file_name), roi_crop.shape)?;
                    for stack_t in 0..index.time_count {
                        if (control.is_cancelled)() {
                            return Err(PredictFailure::Cancelled);
                        }
                        let frame =
                            roi_frame_2d(&stack, &index.axis_order, stack_t, signal_channel, 0)?;
                        let source_t = index.time_indices[stack_t as usize];
                        if let Some(full) = pending.push(FrameBatchItem {
                            pos: *position,
                            sample: sample.name.clone(),
                            signal_channel,
                            roi: roi_crop.roi,
                            t: source_t,
                            width: frame.width,
                            height: frame.height,
                            pixels: frame.into_vec(),
                        }) {
                            commit_batch(full)?;
                        }
                    }
                }
            }
        }
    }
    if let Some(tail) = pending.drain_remainder() {
        commit_batch(tail)?;
    }

    if (control.is_cancelled)() {
        return Err(PredictFailure::Cancelled);
    }

    let traces_dir = output_workspace.join("traces");
    fs::create_dir_all(&traces_dir).map_err(|error| error.to_string())?;
    for ((position, signal_channel), mut rows) in traces_by_pos_channel {
        rows.sort_by_key(|row| (row.pos, row.roi, row.t));
        let output = traces_dir
            .join(format!("Pos{position}"))
            .join(format!("ch{signal_channel}.csv"));
        write_trace_csv(&output, &rows)?;
    }

    prediction_rows.sort_by(|left, right| {
        left.pos
            .cmp(&right.pos)
            .then_with(|| left.sample.cmp(&right.sample))
            .then_with(|| left.roi.cmp(&right.roi))
            .then_with(|| left.t.cmp(&right.t))
    });

    let results_dir = output_workspace.join("results");
    fs::create_dir_all(&results_dir).map_err(|error| error.to_string())?;
    let prediction_csv_rows = prediction_rows
        .iter()
        .map(|row| {
            vec![
                row.t.to_string(),
                row.roi.to_string(),
                format_float(row.p_dead),
                row.label.to_string().to_lowercase(),
                row.pos.to_string(),
                row.sample.clone(),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(
        &results_dir.join("predictions.csv"),
        &["t", "crop", "p_dead", "label", "pos", "sample"],
        &prediction_csv_rows,
    )?;
    Ok(())
}

#[derive(Debug, Clone)]
struct TraceRow {
    pos: u32,
    roi: u32,
    t: u32,
    p_dead: f64,
}

#[derive(Debug, Clone)]
struct PredictionRow {
    t: u32,
    roi: u32,
    p_dead: f64,
    label: bool,
    pos: u32,
    sample: String,
}

fn build_kill_session(model_path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| format!("failed to load kill model: {error}"))
}

fn count_position_frames(
    workspace: &Path,
    signal_channel: u32,
    position: u32,
) -> Result<u32, PredictFailure> {
    let pos_dir = match position_dir(workspace, position) {
        Ok(path) => path,
        Err(_) => return Ok(0),
    };
    let index = read_position_index(&pos_dir)?;
    validate_channel_index(&index, signal_channel)?;
    let count = index.rois.len().saturating_mul(index.time_count as usize);
    u32::try_from(count).map_err(|error| PredictFailure::Failed(error.to_string()))
}

fn normalize_frame(data: &[f64]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in data {
        min = min.min(value);
        max = max.max(value);
    }
    let range = max - min;
    data.iter()
        .map(|&value| {
            if range > 0.0 {
                (((value - min) / range) * 255.0).round() as u8
            } else {
                0
            }
        })
        .collect()
}

fn binary_logits(logits: &ArrayView<f32, ndarray::IxDyn>) -> Result<(f32, f32), String> {
    match logits.ndim() {
        1 if logits.len() >= 2 => Ok((logits[[0]], logits[[1]])),
        2 if logits.shape()[1] >= 2 => Ok((logits[[0, 0]], logits[[0, 1]])),
        4 => Ok((logits[[0, 0, 0, 0]], logits[[0, 1, 0, 0]])),
        _ => Err(format!(
            "unsupported binary classifier logits shape: {:?}",
            logits.shape()
        )),
    }
}

fn resize_to_224(data: &[u8], width: u32, height: u32) -> Result<GrayImage, String> {
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| format!("image dimensions overflow: {width}x{height}"))?;
    if data.len() != expected_len {
        return Err(format!(
            "image buffer length mismatch for {width}x{height}: expected {expected_len}, got {}",
            data.len()
        ));
    }
    let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, data.to_vec())
        .ok_or_else(|| format!("failed to construct image buffer for {width}x{height}"))?;
    Ok(image::imageops::resize(
        &image,
        IMAGE_SIZE,
        IMAGE_SIZE,
        FilterType::Triangle,
    ))
}

fn to_nchw_normalized(gray: &GrayImage) -> Vec<f32> {
    let plane_len = (IMAGE_SIZE * IMAGE_SIZE) as usize;
    let mut output = vec![0.0f32; 3 * plane_len];
    for channel in 0..3 {
        let offset = channel * plane_len;
        for (index, value) in gray.as_raw().iter().enumerate() {
            let normalized = *value as f32 / 255.0;
            output[offset + index] = (normalized - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
        }
    }
    output
}

fn run_batch_inference(
    session: &mut Session,
    input_name: &str,
    batch: &[FrameBatchItem],
) -> Result<Vec<f64>, String> {
    let batch_len = batch.len();
    let mut batch_data = vec![0.0f32; batch_len * 3 * IMAGE_SIZE as usize * IMAGE_SIZE as usize];

    for (index, frame) in batch.iter().enumerate() {
        let normalized = normalize_frame(&frame.pixels);
        let resized = resize_to_224(&normalized, frame.width as u32, frame.height as u32)?;
        let nchw = to_nchw_normalized(&resized);
        let offset = index * 3 * IMAGE_SIZE as usize * IMAGE_SIZE as usize;
        batch_data[offset..offset + nchw.len()].copy_from_slice(&nchw);
    }

    let shape: Ix4 = ndarray::Dim([batch_len, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize]);
    let array = Array::from_shape_vec(shape, batch_data).map_err(|error| error.to_string())?;
    let input_tensor = Tensor::from_array(array).map_err(|error| error.to_string())?;
    let input = ort::inputs![input_name => input_tensor];
    let outputs = session.run(input).map_err(|error| error.to_string())?;
    let logits = if let Some(output) = outputs.get("logits") {
        output.try_extract_array::<f32>()
    } else {
        outputs[0].try_extract_array::<f32>()
    }
    .map_err(|error| error.to_string())?;

    let ndim = logits.ndim();

    let predictions = (0..batch_len)
        .map(|index| {
            let (absent_logit, present_logit) = if ndim == 2 {
                let view = logits.index_axis(Axis(0), index).into_dyn();
                binary_logits(&view)?
            } else {
                (logits[[index, 0, 0, 0]], logits[[index, 1, 0, 0]])
            };
            Ok(super::dead_probability(absent_logit, present_logit))
        })
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(predictions)
}

fn write_trace_csv(path: &Path, rows: &[TraceRow]) -> Result<(), String> {
    let headers = ["roi", "t", "p_dead"];
    let csv_rows = rows
        .iter()
        .map(|row| {
            vec![
                row.roi.to_string(),
                row.t.to_string(),
                format_float(row.p_dead),
            ]
        })
        .collect::<Vec<_>>();
    write_csv_only(path, &headers, &csv_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_to_224_rejects_mismatched_buffer_lengths() {
        let error = resize_to_224(&[0; 3], 2, 2).unwrap_err();
        assert!(error.contains("expected 4, got 3"));
    }

    #[test]
    fn binary_logits_supports_common_export_shapes() {
        let flat_array = ndarray::array![1.0, 2.0];
        assert_eq!(
            binary_logits(&flat_array.view().into_dyn()).unwrap(),
            (1.0, 2.0)
        );

        let batched_array = ndarray::array![[0.5, 1.5]];
        assert_eq!(
            binary_logits(&batched_array.view().into_dyn()).unwrap(),
            (0.5, 1.5)
        );
    }
}

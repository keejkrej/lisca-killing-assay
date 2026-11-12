//! Death-reporter viability, per-cell `t_death` calling, and survival curves.
//!
//! Each ROI crop in `killing-death-reporter` has a background-corrected
//! fluorescence time course (`analysis/Pos{n}/ch{m}.csv`, column `corrected`).
//! Membrane permeabilization (e.g. propidium iodide uptake) produces a sharp,
//! sustained upward step in fluorescence. Pooling all ROIs across the workspace
//! calibrates the live baseline and frame-to-frame noise floor so per-cell
//! `t_death` calls and sample `% viable` survival curves need no classifier.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::array::quantile;
use crate::csv_io::{column_index, format_float, parse_f64, read_csv, write_csv_only};
use crate::export::write_xlsx_only;
use crate::sample::{sample_pack_dirnames, SampleMapping};

#[derive(Debug, Clone)]
struct RoiSeries {
    sample_idx: usize,
    pos: u32,
    roi: u32,
    frames: Vec<(u32, f64)>,
}

#[derive(Debug, Clone)]
struct RoiViabilityCall {
    sample_idx: usize,
    pos: u32,
    roi: u32,
    dead: bool,
    death_frame: Option<u32>,
    baseline: f64,
    peak: f64,
    snr: f64,
}

/// Compute per-ROI death calls and sample survival curves from measured
/// `analysis/Pos{n}/ch{m}.csv` fluorescence series.
pub fn run_viability(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
) -> Result<(), String> {
    if interval <= 0.0 || !interval.is_finite() {
        return Err(format!("interval must be > 0, got {interval}"));
    }

    let series = load_workspace_series(workspace, mapping)?;
    if series.is_empty() {
        return Err("no death-reporter fluorescence traces found to score viability".to_string());
    }

    let calls = call_workspace_viability(&series);
    let dirnames = sample_pack_dirnames(mapping);

    // 1. Write `analysis/Pos{pos}/viability.csv` per position.
    let mut by_pos: BTreeMap<u32, Vec<&RoiViabilityCall>> = BTreeMap::new();
    for call in &calls {
        by_pos.entry(call.pos).or_default().push(call);
    }
    for (pos, pos_calls) in &by_pos {
        let mut rows = Vec::with_capacity(pos_calls.len());
        for call in pos_calls {
            let death_time_str = call
                .death_frame
                .map(|frame| frame.to_string())
                .unwrap_or_default();
            let death_min_str = call
                .death_frame
                .map(|frame| format_float(f64::from(frame) * interval))
                .unwrap_or_default();
            rows.push(vec![
                call.roi.to_string(),
                call.dead.to_string(),
                death_time_str,
                death_min_str,
                format_float(call.baseline),
                format_float(call.peak),
                format_float(call.snr),
            ]);
        }
        write_csv_only(
            &workspace.join(format!("analysis/Pos{pos}/viability.csv")),
            &[
                "roi",
                "dead",
                "death_time",
                "death_minutes",
                "baseline",
                "peak",
                "snr",
            ],
            &rows,
        )?;
    }

    // 2. Write `results/<sample>/viability.xlsx` and `results/death_times.csv`.
    let mut death_time_rows = Vec::with_capacity(calls.len());
    let mut sample_xlsx_rows: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    for call in &calls {
        let sample_name = mapping
            .get(call.sample_idx)
            .map(|entry| entry.name.as_str())
            .unwrap_or("sample");
        let death_time_str = call
            .death_frame
            .map(|frame| frame.to_string())
            .unwrap_or_default();
        let death_min_str = call
            .death_frame
            .map(|frame| format_float(f64::from(frame) * interval))
            .unwrap_or_default();
        death_time_rows.push(vec![
            call.pos.to_string(),
            call.roi.to_string(),
            sample_name.to_string(),
            call.dead.to_string(),
            death_time_str.clone(),
            death_min_str.clone(),
            format_float(call.baseline),
            format_float(call.peak),
            format_float(call.snr),
        ]);
        sample_xlsx_rows
            .entry(call.sample_idx)
            .or_default()
            .push(vec![
                call.pos.to_string(),
                call.roi.to_string(),
                call.dead.to_string(),
                death_time_str,
                death_min_str,
                format_float(call.baseline),
                format_float(call.peak),
                format_float(call.snr),
            ]);
    }
    write_csv_only(
        &workspace.join("results/death_times.csv"),
        &[
            "pos",
            "roi",
            "sample",
            "dead",
            "death_time",
            "death_minutes",
            "baseline",
            "peak",
            "snr",
        ],
        &death_time_rows,
    )?;
    for (sample_idx, rows) in &sample_xlsx_rows {
        let dirname = dirnames
            .get(sample_idx)
            .map(String::as_str)
            .unwrap_or("sample");
        write_xlsx_only(
            &workspace
                .join("results")
                .join(dirname)
                .join("viability.xlsx"),
            &[
                "pos",
                "roi",
                "dead",
                "death_time",
                "death_minutes",
                "baseline",
                "peak",
                "snr",
            ],
            rows,
        )?;
    }

    // 3. Write `results/kill_curve.csv` with both `n_alive` and `pct_viable`.
    let mut frames_by_sample: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    for item in &series {
        let entry = frames_by_sample.entry(item.sample_idx).or_default();
        for &(t, _) in &item.frames {
            entry.insert(t);
        }
    }
    let mut calls_by_sample: BTreeMap<usize, Vec<&RoiViabilityCall>> = BTreeMap::new();
    for call in &calls {
        calls_by_sample
            .entry(call.sample_idx)
            .or_default()
            .push(call);
    }

    let mut curve_rows = Vec::new();
    for (sample_idx, sample) in mapping.iter().enumerate() {
        let Some(sample_calls) = calls_by_sample.get(&sample_idx) else {
            continue;
        };
        let Some(timepoints) = frames_by_sample.get(&sample_idx) else {
            continue;
        };
        let n_total = sample_calls.len();
        if n_total == 0 {
            continue;
        }
        for &t in timepoints {
            let n_dead = sample_calls
                .iter()
                .filter(|call| call.dead && call.death_frame.is_some_and(|df| df <= t))
                .count();
            let n_alive = n_total.saturating_sub(n_dead);
            let pct_viable = (n_alive as f64) * 100.0 / (n_total as f64);
            let pct_dead = 100.0 - pct_viable;
            curve_rows.push(vec![
                sample.name.clone(),
                t.to_string(),
                format_float(f64::from(t) * interval),
                n_total.to_string(),
                n_alive.to_string(),
                n_dead.to_string(),
                format_float(pct_viable),
                format_float(pct_dead),
            ]);
        }
    }
    write_csv_only(
        &workspace.join("results/kill_curve.csv"),
        &[
            "sample",
            "t",
            "minutes",
            "n_total",
            "n_alive",
            "n_dead",
            "pct_viable",
            "pct_dead",
        ],
        &curve_rows,
    )?;

    Ok(())
}

fn load_workspace_series(
    workspace: &Path,
    mapping: &SampleMapping,
) -> Result<Vec<RoiSeries>, String> {
    let mut series = Vec::new();
    for (sample_idx, sample) in mapping.iter().enumerate() {
        let Some(&signal_ch) = sample.signal.first() else {
            return Err(format!("sample '{}' has no signal channel", sample.name));
        };
        for &pos in &sample.positions {
            let csv_path = workspace.join(format!("analysis/Pos{pos}/ch{signal_ch}.csv"));
            if !csv_path.is_file() {
                continue;
            }
            let (headers, rows) = read_csv(&csv_path)?;
            let roi_idx =
                column_index(&headers, "roi").ok_or("missing roi column in channel csv")?;
            let t_idx = column_index(&headers, "t").ok_or("missing t column in channel csv")?;
            let val_idx = column_index(&headers, "corrected")
                .or_else(|| column_index(&headers, "sum"))
                .ok_or("missing corrected/sum column in channel csv")?;

            let mut by_roi: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
            for row in rows {
                let Some(roi) = row.get(roi_idx).and_then(|s| s.trim().parse::<u32>().ok()) else {
                    continue;
                };
                let Some(t) = row.get(t_idx).and_then(|s| s.trim().parse::<u32>().ok()) else {
                    continue;
                };
                let Some(value) = row.get(val_idx).and_then(|s| parse_f64(s)) else {
                    continue;
                };
                if value.is_finite() {
                    by_roi.entry(roi).or_default().push((t, value));
                }
            }
            for (roi, mut frames) in by_roi {
                frames.sort_by_key(|&(t, _)| t);
                frames.dedup_by_key(|entry| entry.0);
                if !frames.is_empty() {
                    series.push(RoiSeries {
                        sample_idx,
                        pos,
                        roi,
                        frames,
                    });
                }
            }
        }
    }
    Ok(series)
}

fn diff_mad_sigma(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let diffs: Vec<f64> = values
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .collect();
    quantile(&diffs, 0.5) / 0.9539
}

fn segment_sse(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    values
        .iter()
        .map(|&value| {
            let d = value - mean;
            d * d
        })
        .sum()
}

fn call_workspace_viability(series: &[RoiSeries]) -> Vec<RoiViabilityCall> {
    let mut early_baselines = Vec::with_capacity(series.len());
    let mut positive_sigmas = Vec::new();
    let mut per_roi_sigmas = Vec::with_capacity(series.len());

    for item in series {
        let values: Vec<f64> = item.frames.iter().map(|&(_, v)| v).collect();
        let early_len = values.len().min(3);
        let early_med = quantile(&values[..early_len], 0.5);
        early_baselines.push(early_med);
        let sigma = diff_mad_sigma(&values);
        per_roi_sigmas.push(sigma);
        if sigma > 1e-9 {
            positive_sigmas.push(sigma);
        }
    }

    let assay_live_baseline = quantile(&early_baselines, 0.25);
    let assay_sigma = if positive_sigmas.is_empty() {
        (assay_live_baseline.abs() * 0.05).max(1.0)
    } else {
        quantile(&positive_sigmas, 0.5)
    };
    let min_step = (6.0 * assay_sigma)
        .max(assay_live_baseline.abs() * 0.5)
        .max(10.0);
    let allow_t0_dead = series.len() >= 4;

    series
        .iter()
        .zip(per_roi_sigmas)
        .map(|(item, roi_sigma)| {
            let values: Vec<f64> = item.frames.iter().map(|&(_, v)| v).collect();
            let early_len = values.len().min(3);
            let early_med = quantile(&values[..early_len], 0.5);
            let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let noise = roi_sigma.max(assay_sigma).max(1.0);

            if values.len() < 2 {
                return RoiViabilityCall {
                    sample_idx: item.sample_idx,
                    pos: item.pos,
                    roi: item.roi,
                    dead: false,
                    death_frame: None,
                    baseline: early_med,
                    peak,
                    snr: 0.0,
                };
            }

            let mut best_split: Option<(usize, f64, f64, f64)> = None;
            for k in 1..values.len() {
                let pre = &values[..k];
                let post = &values[k..];
                let pre_med = quantile(pre, 0.5);
                let post_med = quantile(post, 0.5);
                let step = post_med - pre_med;
                let threshold = min_step.max(6.0 * roi_sigma).max(pre_med.abs() * 0.5);
                if step < threshold {
                    continue;
                }
                let sustain_end = (k + 3).min(values.len());
                let sustain_med = quantile(&values[k..sustain_end], 0.5);
                if sustain_med - pre_med < threshold * 0.8 {
                    continue;
                }
                let residual_sse = segment_sse(pre) + segment_sse(post);
                match best_split {
                    None => best_split = Some((k, residual_sse, pre_med, step)),
                    Some((_, best_sse, _, _)) if residual_sse < best_sse => {
                        best_split = Some((k, residual_sse, pre_med, step));
                    }
                    _ => {}
                }
            }

            if let Some((k, _, pre_med, step)) = best_split {
                let pre_sigma = diff_mad_sigma(&values[..k]).max(assay_sigma).max(1.0);
                return RoiViabilityCall {
                    sample_idx: item.sample_idx,
                    pos: item.pos,
                    roi: item.roi,
                    dead: true,
                    death_frame: Some(item.frames[k].0),
                    baseline: pre_med,
                    peak,
                    snr: (step / pre_sigma).max(0.0),
                };
            }

            if allow_t0_dead && values.len() >= 3 {
                let median_all = quantile(&values, 0.5);
                let step_over_live = median_all - assay_live_baseline;
                if early_med >= assay_live_baseline + min_step * 2.0
                    && step_over_live >= min_step * 2.0
                {
                    return RoiViabilityCall {
                        sample_idx: item.sample_idx,
                        pos: item.pos,
                        roi: item.roi,
                        dead: true,
                        death_frame: Some(item.frames[0].0),
                        baseline: assay_live_baseline,
                        peak,
                        snr: (step_over_live / noise).max(0.0),
                    };
                }
            }

            let snr = ((peak - early_med) / noise).max(0.0);
            RoiViabilityCall {
                sample_idx: item.sample_idx,
                pos: item.pos,
                roi: item.roi,
                dead: false,
                death_frame: None,
                baseline: early_med,
                peak,
                snr,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SampleAnalysis;

    fn write_trace_csv(workspace: &Path, pos: u32, channel: u32, traces: &[(u32, &[f64])]) {
        let mut rows = Vec::new();
        for &(roi, values) in traces {
            for (t, &val) in values.iter().enumerate() {
                rows.push(vec![
                    roi.to_string(),
                    t.to_string(),
                    "100".to_string(),
                    "10".to_string(),
                    format_float(val + 1000.0),
                    format_float(val),
                ]);
            }
        }
        write_csv_only(
            &workspace.join(format!("analysis/Pos{pos}/ch{channel}.csv")),
            &["roi", "t", "area", "background", "sum", "corrected"],
            &rows,
        )
        .unwrap();
    }

    #[test]
    fn calls_pi_step_death_times_and_writes_survival_curve() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        write_trace_csv(
            root,
            1,
            1,
            &[
                (0, &[5.0, 6.0, 4.5, 5.5, 5.0, 6.0]),
                (1, &[8.0, 7.0, 8.5, 7.5, 8.0, 7.0]),
            ],
        );
        write_trace_csv(
            root,
            2,
            1,
            &[
                (0, &[6.0, 7.0, 420.0, 430.0, 415.0, 410.0]),
                (1, &[5.0, 6.0, 5.5, 6.5, 380.0, 390.0]),
                (2, &[7.0, 6.5, 7.5, 6.0, 7.0, 6.5]),
                (3, &[450.0, 460.0, 445.0, 455.0, 450.0, 440.0]),
            ],
        );

        let mapping = SampleMapping(vec![
            SampleAnalysis {
                name: "Control".into(),
                positions: vec![1],
                signal: vec![1],
                segmentation: 0,
            },
            SampleAnalysis {
                name: "Effector 5:1".into(),
                positions: vec![2],
                signal: vec![1],
                segmentation: 0,
            },
        ]);

        run_viability(root, &mapping, 5.0).unwrap();

        let (_, death_rows) = read_csv(&root.join("results/death_times.csv")).unwrap();
        assert_eq!(death_rows.len(), 6);
        assert_eq!(death_rows[0][3], "false");
        assert_eq!(death_rows[0][4], "");
        assert_eq!(death_rows[1][3], "false");
        assert_eq!(death_rows[2][3], "true");
        assert_eq!(death_rows[2][4], "2");
        assert_eq!(death_rows[2][5], "10");
        assert_eq!(death_rows[3][3], "true");
        assert_eq!(death_rows[3][4], "4");
        assert_eq!(death_rows[3][5], "20");
        assert_eq!(death_rows[4][3], "false");
        assert_eq!(death_rows[5][3], "true");
        assert_eq!(death_rows[5][4], "0");

        let (curve_headers, curve_rows) = read_csv(&root.join("results/kill_curve.csv")).unwrap();
        assert_eq!(
            curve_headers,
            [
                "sample",
                "t",
                "minutes",
                "n_total",
                "n_alive",
                "n_dead",
                "pct_viable",
                "pct_dead"
            ]
        );
        let control_t5 = curve_rows
            .iter()
            .find(|r| r[0] == "Control" && r[1] == "5")
            .unwrap();
        assert_eq!(control_t5[4], "2");
        assert_eq!(control_t5[6], "100");

        let treated_t0 = curve_rows
            .iter()
            .find(|r| r[0] == "Effector 5:1" && r[1] == "0")
            .unwrap();
        assert_eq!(treated_t0[6], "75");
        let treated_t2 = curve_rows
            .iter()
            .find(|r| r[0] == "Effector 5:1" && r[1] == "2")
            .unwrap();
        assert_eq!(treated_t2[6], "50");
        let treated_t4 = curve_rows
            .iter()
            .find(|r| r[0] == "Effector 5:1" && r[1] == "4")
            .unwrap();
        assert_eq!(treated_t4[6], "25");

        assert!(root.join("analysis/Pos1/viability.csv").is_file());
        assert!(root.join("analysis/Pos2/viability.csv").is_file());
        assert!(root.join("results/Control/viability.xlsx").is_file());
        assert!(root.join("results/Effector_5_1/viability.xlsx").is_file());
    }
}

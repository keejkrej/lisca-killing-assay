//! Figures for killing assays (`killing-death-reporter` and `killing-engagement`).

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mplot::prelude::{
    AxesStyle, Figure, FigureBuilder, FillBetweenStyle, GridPos, HistStyle, LegendStyle, LineDash,
    LineStyle, SaveOptions, Size, TickFormat,
};
use mplot::Color;

use crate::array::{percentile, quantile};
use crate::csv_io::{column_index, parse_f64, read_csv};
use crate::engagement::discover_engagement_csvs;
use crate::sample::{sample_pack_dirnames, SampleMapping};
use crate::traces_pack::{
    discover_trace_csvs, group_trace_rows, load_trace_panels_by_sample, publish_sample_traces_xlsx,
    TracePanel,
};

const FIGURE_SINGLE_WIDTH_IN: f64 = 6.5;
const FIGURE_SINGLE_HEIGHT_IN: f64 = 5.5;
const PANEL_WIDTH_IN: f64 = 5.5;
const PANEL_HEIGHT_IN: f64 = 4.0;
const FIGURE_DPI: u32 = 100;
const LABEL_FONTSIZE: f64 = 18.0;
const TITLE_FONTSIZE: f64 = 18.0;
const TICK_FONTSIZE: f64 = 17.0;
const SAVE_PAD_GRID_INCHES: f64 = 0.28;
const SAVE_PAD_SINGLE_INCHES: f64 = 0.25;
const HISTOGRAM_BINS: usize = 20;

const SURVIVAL_PALETTE: [&str; 10] = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
];

/// Death-reporter fluorescence trace plots + `results/<sample>/traces.xlsx`.
pub fn run_plot_traces(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
    columns: Option<usize>,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Err(format!("interval must be > 0, got {interval}"));
    }
    let _ = columns;
    let csvs = discover_trace_csvs(&workspace.join("analysis"))?;
    let panels = load_trace_panels_by_sample(&csvs, "corrected", mapping)?;
    if panels.is_empty() {
        return Err("no fluorescence trace panels to plot".to_string());
    }

    publish_sample_traces_xlsx(workspace, mapping)?;
    let dirnames = sample_pack_dirnames(mapping);
    write_per_sample_metric_plots(
        &panels,
        |sample| {
            let dirname = dirnames
                .get(&sample)
                .map(String::as_str)
                .unwrap_or("sample");
            workspace.join("results").join(dirname)
        },
        "traces",
        "fluorescence",
        interval,
        mapping,
    )
}

/// Fluorescent-engagement count trace plots (`engagement_traces*.png`).
pub fn run_plot_engagement_traces(
    workspace: &Path,
    mapping: &SampleMapping,
    interval_minutes: f64,
) -> Result<(), String> {
    if interval_minutes <= 0.0 {
        return Err(format!("interval must be > 0, got {interval_minutes}"));
    }
    let csvs = discover_engagement_csvs(&workspace.join("analysis"))?;
    let panels = load_engagement_panels(&csvs, "t_cells", mapping)?;
    if panels.is_empty() {
        return Err("no engagement traces to plot".to_string());
    }
    let dirnames = sample_pack_dirnames(mapping);
    write_per_sample_metric_plots(
        &panels,
        |sample| {
            let dirname = dirnames
                .get(&sample)
                .map(String::as_str)
                .unwrap_or("sample");
            workspace.join("results").join(dirname)
        },
        "engagement_traces",
        "engagers",
        interval_minutes,
        mapping,
    )
}

/// Combined sample survival curve (`results/survival_curve.png`).
pub fn run_plot_survival(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Err(format!("interval must be > 0, got {interval}"));
    }

    let curve_csv = workspace.join("results/kill_curve.csv");
    let (headers, rows) = read_csv(&curve_csv)?;
    let t_index = column_index(&headers, "t").ok_or("missing t in kill_curve.csv")?;
    let sample_index =
        column_index(&headers, "sample").ok_or("missing sample in kill_curve.csv")?;
    let pct_index = column_index(&headers, "pct_viable");
    let alive_index = column_index(&headers, "n_alive");
    let total_index = column_index(&headers, "n_total");

    if pct_index.is_none() && alive_index.is_none() {
        return Err("missing pct_viable or n_alive in kill_curve.csv".to_string());
    }

    let labels = sample_labels(mapping);
    let mut grouped_raw: BTreeMap<usize, Vec<(f64, f64, Option<f64>)>> = BTreeMap::new();
    for row in rows {
        let Some(t) = row.get(t_index).and_then(|v| parse_f64(v)) else {
            continue;
        };
        let Some(sample_name) = row.get(sample_index) else {
            continue;
        };
        let Some(sample) = mapping.iter().position(|entry| &entry.name == sample_name) else {
            continue;
        };
        let pct_opt = pct_index
            .and_then(|idx| row.get(idx))
            .and_then(|v| parse_f64(v));
        let alive = alive_index
            .and_then(|idx| row.get(idx))
            .and_then(|v| parse_f64(v))
            .unwrap_or(0.0);
        let total_opt = total_index
            .and_then(|idx| row.get(idx))
            .and_then(|v| parse_f64(v));
        let y = if let Some(pct) = pct_opt {
            pct
        } else if let Some(total) = total_opt.filter(|&tot| tot > 0.0) {
            (alive * 100.0) / total
        } else {
            alive
        };
        grouped_raw
            .entry(sample)
            .or_default()
            .push((t * interval, y, pct_opt.or(total_opt)));
    }

    if grouped_raw.is_empty() {
        return Err("no survival curve points to plot".to_string());
    }

    let mut grouped: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
    for (sample, mut pts) in grouped_raw {
        pts.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
        let has_explicit_pct = pts.iter().any(|p| p.2.is_some());
        let denom = if has_explicit_pct {
            1.0
        } else {
            pts.iter().map(|p| p.1).fold(0.0_f64, f64::max)
        };
        let normalized = pts
            .into_iter()
            .map(|(x, y, _)| {
                let pct = if has_explicit_pct {
                    y
                } else if denom > 0.0 {
                    (y * 100.0) / denom
                } else {
                    100.0
                };
                (x, pct.clamp(0.0, 100.0))
            })
            .collect();
        grouped.insert(sample, normalized);
    }

    let max_x = grouped
        .values()
        .flat_map(|pts| pts.iter().map(|p| p.0))
        .fold(0.0_f64, f64::max)
        .max(interval);

    let builder = figure_builder_single().panel(GridPos::new(1, 1, 1), move |panel| {
        for (&sample, points) in &grouped {
            let color = SURVIVAL_PALETTE[sample % SURVIVAL_PALETTE.len()];
            let label = labels
                .get(&sample)
                .cloned()
                .unwrap_or_else(|| format!("Sample {}", sample + 1));
            let x: Vec<f64> = points.iter().map(|p| p.0).collect();
            let y: Vec<f64> = points.iter().map(|p| p.1).collect();
            panel.line(&x, &y, trace_line_style(color, 1.0).width(2.0).label(label));
        }
        panel.axes(
            AxesStyle::new()
                .title("Cell viability")
                .x_label("time (min)")
                .y_label("viable cells (%)")
                .x_range(0.0, max_x)
                .y_range(0.0, 105.0)
                .legend(LegendStyle::show()),
        );
    });

    let figure = builder.build().map_err(|error| error.to_string())?;
    save_figure(
        &figure,
        &workspace.join("results/survival_curve.png"),
        SAVE_PAD_SINGLE_INCHES,
    )
}

/// Per-sample `N(alive)` kill curve grid (`results/kill_curve.png`).
pub fn run_plot_kill(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Err(format!("interval must be > 0, got {interval}"));
    }

    let curve_csv = workspace.join("results/kill_curve.csv");
    let (headers, rows) = read_csv(&curve_csv)?;
    let t_index = column_index(&headers, "t").ok_or("missing t in kill_curve.csv")?;
    let alive_index =
        column_index(&headers, "n_alive").ok_or("missing n_alive in kill_curve.csv")?;
    let sample_index =
        column_index(&headers, "sample").ok_or("missing sample in kill_curve.csv")?;

    let labels = sample_labels(mapping);
    let mut grouped: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
    for row in rows {
        let Some(t) = parse_f64(&row[t_index]) else {
            continue;
        };
        let Some(n_alive) = parse_f64(&row[alive_index]) else {
            continue;
        };
        let Some(sample) = mapping
            .iter()
            .position(|entry| entry.name == row[sample_index])
        else {
            continue;
        };
        grouped
            .entry(sample)
            .or_default()
            .push((t * interval, n_alive));
    }

    if grouped.is_empty() {
        return Err("no kill curve points to plot".to_string());
    }

    for points in grouped.values_mut() {
        points.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    }

    let samples: Vec<usize> = grouped.keys().copied().collect();
    let (rows, cols) = grid_dimensions(samples.len(), 2);
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, sample) in samples.iter().enumerate() {
        let points = grouped.get(sample).cloned().unwrap_or_default();
        let label = labels.get(sample).cloned().unwrap_or_default();
        let max_x = points.iter().map(|point| point.0).fold(0.0f64, f64::max);
        let max_y = points.iter().map(|point| point.1).fold(0.0f64, f64::max);

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |panel| {
            let x: Vec<f64> = points.iter().map(|point| point.0).collect();
            let y: Vec<f64> = points.iter().map(|point| point.1).collect();
            panel.line(&x, &y, trace_line_style("steelblue", 1.0));
            panel.axes(
                AxesStyle::new()
                    .title(label)
                    .x_label("time (min)")
                    .y_label("N(alive)")
                    .x_range(0.0, max_x.max(interval))
                    .y_range(0.0, max_y.max(1.0)),
            );
        });
    }

    for index in samples.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    save_figure(
        &figure,
        &workspace.join("results/kill_curve.png"),
        SAVE_PAD_GRID_INCHES,
    )
}

/// Per-sample death-time histogram (`results/death_times.png`).
pub fn run_plot_death_times(
    workspace: &Path,
    mapping: &SampleMapping,
    interval: f64,
) -> Result<(), String> {
    if interval <= 0.0 {
        return Err(format!("interval must be > 0, got {interval}"));
    }

    let death_csv = workspace.join("results/death_times.csv");
    let (headers, rows) = read_csv(&death_csv)?;
    let death_time_index =
        column_index(&headers, "death_time").ok_or("missing death_time in death_times.csv")?;
    let sample_index =
        column_index(&headers, "sample").ok_or("missing sample in death_times.csv")?;
    let dead_index = column_index(&headers, "dead");

    let labels = sample_labels(mapping);
    let mut grouped: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for row in rows {
        if let Some(idx) = dead_index {
            if row.get(idx).map(|s| s.trim()) != Some("true") {
                continue;
            }
        }
        let Some(death_time) = row.get(death_time_index).and_then(|v| parse_f64(v)) else {
            continue;
        };
        if dead_index.is_some() {
            if death_time < 0.0 {
                continue;
            }
        } else if death_time <= 0.0 {
            continue;
        }
        let Some(sample_name) = row.get(sample_index) else {
            continue;
        };
        let Some(sample) = mapping.iter().position(|entry| &entry.name == sample_name) else {
            continue;
        };
        grouped
            .entry(sample)
            .or_default()
            .push(death_time * interval);
    }

    if grouped.is_empty() {
        let _ = std::fs::remove_file(workspace.join("results/death_times.png"));
        return Ok(());
    }

    let samples: Vec<usize> = grouped.keys().copied().collect();
    let (rows, cols) = grid_dimensions(samples.len(), 2);
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, sample) in samples.iter().enumerate() {
        let values = grouped.get(sample).cloned().unwrap_or_default();
        let label = labels.get(sample).cloned().unwrap_or_default();
        let max_count = values.len() as f64;
        let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (x_min, x_max) = if !min_v.is_finite() || !max_v.is_finite() {
            (0.0, interval.max(1.0))
        } else if (max_v - min_v).abs() < 1e-12 {
            let pad = interval.max(1.0);
            ((min_v - pad).max(0.0), min_v + pad.max(1.0))
        } else {
            let pad = (max_v - min_v) * 0.05;
            ((min_v - pad).max(0.0), max_v + pad)
        };

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |panel| {
            panel.histogram(
                &values,
                HistStyle::new()
                    .bins(HISTOGRAM_BINS)
                    .color(Color::hex("steelblue"))
                    .label("n crops"),
            );
            panel.axes(
                AxesStyle::new()
                    .title(label)
                    .x_label("time (min)")
                    .y_label("n crops")
                    .x_range(x_min, x_max)
                    .y_range(0.0, max_count.max(1.0)),
            );
        });
    }

    for index in samples.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    save_figure(
        &figure,
        &workspace.join("results/death_times.png"),
        SAVE_PAD_GRID_INCHES,
    )
}

fn load_engagement_panels(
    csvs: &[PathBuf],
    y_column: &str,
    mapping: &SampleMapping,
) -> Result<Vec<TracePanel>, String> {
    let mut grouped = BTreeMap::<usize, TracePanel>::new();
    for path in csvs {
        let position = engagement_position(path)?;
        let sample = mapping
            .iter()
            .position(|sample| sample.positions.contains(&position))
            .ok_or_else(|| format!("No sample owns Pos{position} ({})", path.display()))?;
        let (headers, rows) = read_csv(path)?;
        let groups = group_trace_rows(&headers, &rows, y_column)?;
        let mut y_values = Vec::new();
        let traces = groups
            .into_values()
            .map(|mut points| {
                points
                    .sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
                y_values.extend(points.iter().map(|(_, value)| *value));
                points
            })
            .collect::<Vec<_>>();
        let entry = grouped.entry(sample).or_insert_with(|| TracePanel {
            sample,
            paths: Vec::new(),
            traces: Vec::new(),
            y_values: Vec::new(),
        });
        entry.paths.push(path.clone());
        entry.traces.extend(traces);
        entry.y_values.extend(y_values);
    }
    Ok(grouped.into_values().collect())
}

fn engagement_position(path: &Path) -> Result<u32, String> {
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("Pos"))
        .and_then(|rest| rest.parse().ok())
        .ok_or_else(|| format!("Expected Pos{{n}}/engagement.csv, got {}", path.display()))
}

fn figure_builder_single() -> FigureBuilder {
    figure_builder(FIGURE_SINGLE_WIDTH_IN, FIGURE_SINGLE_HEIGHT_IN, 0.2, 0.2)
}

fn figure_size_for_grid(nrows: usize, ncols: usize) -> (f64, f64) {
    let rows = nrows.max(1);
    let cols = ncols.max(1);
    if rows == 1 && cols == 1 {
        return (FIGURE_SINGLE_WIDTH_IN, FIGURE_SINGLE_HEIGHT_IN);
    }
    let panel_slots = rows * cols;
    let scale = if panel_slots >= 12 {
        0.75
    } else if panel_slots >= 8 {
        0.85
    } else {
        1.0
    };
    (
        PANEL_WIDTH_IN * cols as f64 * scale,
        PANEL_HEIGHT_IN * rows as f64 * scale,
    )
}

fn figure_builder_for_grid(nrows: usize, ncols: usize) -> FigureBuilder {
    let (width_in, height_in) = figure_size_for_grid(nrows, ncols);
    figure_builder(width_in, height_in, 0.16, 0.22)
}

fn figure_builder(width_in: f64, height_in: f64, h_gap: f64, v_gap: f64) -> FigureBuilder {
    Figure::builder()
        .size(Size::inches(width_in, height_in))
        .label_fontsize(LABEL_FONTSIZE)
        .tick_fontsize(TICK_FONTSIZE)
        .title_fontsize(TITLE_FONTSIZE)
        .gaps(h_gap, v_gap)
}

fn trace_line_style(color_name: &str, alpha: f64) -> LineStyle {
    LineStyle::new()
        .color(Color::hex(color_name))
        .alpha(alpha)
        .width(1.5)
}

fn save_figure(figure: &Figure, path: &Path, pad_inches: f64) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    figure
        .save(
            path,
            SaveOptions::new().dpi(FIGURE_DPI).pad_inches(pad_inches),
        )
        .map_err(|error| error.to_string())
}

fn sample_labels(mapping: &SampleMapping) -> BTreeMap<usize, String> {
    mapping
        .iter()
        .enumerate()
        .map(|(index, sample)| (index, sample.name.clone()))
        .collect()
}

fn sample_label(sample: usize, mapping: &SampleMapping) -> String {
    mapping
        .get(sample)
        .map(|entry| entry.name.clone())
        .unwrap_or_else(|| format!("sample {sample}"))
}

fn percentile_ylim(values: &[f64]) -> (f64, f64) {
    percentile_ylim_with(values, 0.0, 100.0, 1.0, 0.9)
}

fn percentile_ylim_with(
    values: &[f64],
    low_percentile: f64,
    high_percentile: f64,
    low_margin: f64,
    high_margin: f64,
) -> (f64, f64) {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    if !(0.0 < low_margin && low_margin <= 1.0 && 0.0 < high_margin && high_margin <= 1.0) {
        return (0.0, 1.0);
    }
    if !(0.0 <= low_percentile && low_percentile < high_percentile && high_percentile <= 100.0) {
        return (0.0, 1.0);
    }
    let low = percentile(&finite, low_percentile) * low_margin;
    let high = percentile(&finite, high_percentile) / high_margin;
    expand_degenerate_ylim(low, high)
}

fn expand_degenerate_ylim(low: f64, high: f64) -> (f64, f64) {
    if !low.is_finite() || !high.is_finite() {
        return (0.0, 1.0);
    }
    if low < high {
        return (low, high);
    }
    let pad = if low == 0.0 { 1.0 } else { low.abs() * 0.05 };
    (low - pad, high + pad)
}

fn subplot_grid_shape(panel_count: usize) -> (usize, usize) {
    let n = panel_count.max(1);
    match n {
        1 => (1, 1),
        2 => (1, 2),
        3 | 4 => (2, 2),
        5 | 6 => (2, 3),
        7..=9 => (3, 3),
        10..=12 => (3, 4),
        _ => {
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = n.div_ceil(cols);
            (rows.max(1), cols.max(1))
        }
    }
}

fn resolve_subplot_grid(panel_count: usize, columns: Option<usize>) -> (usize, usize) {
    match columns {
        None => {
            if panel_count == 0 {
                (1, 1)
            } else {
                subplot_grid_shape(panel_count)
            }
        }
        Some(cols) => {
            let cols = cols.max(1);
            if panel_count == 0 {
                (1, cols)
            } else {
                (panel_count.div_ceil(cols).max(1), cols)
            }
        }
    }
}

fn grid_dimensions(count: usize, columns: usize) -> (usize, usize) {
    resolve_subplot_grid(count, Some(columns))
}

fn sample_subplot_title(sample: usize, trace_count: usize, mapping: &SampleMapping) -> String {
    let label = sample_label(sample, mapping);
    format!("{label} ({trace_count} traces)")
}

fn sample_trace_naming_haystack(
    sample: usize,
    paths: &[PathBuf],
    mapping: &SampleMapping,
) -> String {
    let mut parts = vec![sample_label(sample, mapping)];
    for path in paths {
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            parts.push(name.to_string());
        }
    }
    parts.join(" ")
}

fn trace_color_alpha(haystack: &str) -> (&'static str, f64) {
    let lower = haystack.to_lowercase();
    let color = if lower.contains("egfp") || lower.contains("gfp") {
        "green"
    } else if lower.contains("mcherry") {
        "red"
    } else if lower.contains("yfp") {
        "yellow"
    } else if lower.contains("bfp") {
        "blue"
    } else {
        "gray"
    };
    (color, 0.1)
}

#[derive(Debug, Clone)]
struct SampleSummary {
    t_minutes: Vec<f64>,
    mean: Vec<f64>,
    median: Vec<f64>,
    q25: Vec<f64>,
    q75: Vec<f64>,
    trace_count: usize,
}

fn sample_summary_curves(traces: &[Vec<(f64, f64)>], interval: f64) -> Option<SampleSummary> {
    let mut by_time: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    let mut trace_count = 0usize;
    for trace in traces {
        let mut used = false;
        for &(t, y) in trace {
            if !t.is_finite() || !y.is_finite() {
                continue;
            }
            let minutes = t * interval;
            let key = (minutes * 1_000_000.0).round() as i64;
            by_time.entry(key).or_default().push(y);
            used = true;
        }
        if used {
            trace_count += 1;
        }
    }
    if by_time.is_empty() || trace_count == 0 {
        return None;
    }

    let mut t_minutes = Vec::with_capacity(by_time.len());
    let mut mean = Vec::with_capacity(by_time.len());
    let mut median = Vec::with_capacity(by_time.len());
    let mut q25 = Vec::with_capacity(by_time.len());
    let mut q75 = Vec::with_capacity(by_time.len());
    for (key, values) in by_time {
        t_minutes.push(key as f64 / 1_000_000.0);
        let n = values.len() as f64;
        mean.push(values.iter().sum::<f64>() / n);
        median.push(quantile(&values, 0.5));
        q25.push(quantile(&values, 0.25));
        q75.push(quantile(&values, 0.75));
    }
    Some(SampleSummary {
        t_minutes,
        mean,
        median,
        q25,
        q75,
        trace_count,
    })
}

fn summary_ylim(summary: Option<&SampleSummary>) -> (f64, f64) {
    let Some(summary) = summary else {
        return (0.0, 1.0);
    };
    let mut values = Vec::with_capacity(
        summary.mean.len() + summary.median.len() + summary.q25.len() + summary.q75.len(),
    );
    values.extend_from_slice(&summary.mean);
    values.extend_from_slice(&summary.median);
    values.extend_from_slice(&summary.q25);
    values.extend_from_slice(&summary.q75);
    percentile_ylim(&values)
}

fn companion_plot_path(primary: &Path, suffix: &str) -> PathBuf {
    let stem = primary
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("plot");
    primary.with_file_name(format!("{stem}_{suffix}.png"))
}

pub(crate) fn write_per_sample_metric_plots(
    panels: &[TracePanel],
    sample_dir: impl Fn(usize) -> PathBuf,
    file_stem: &str,
    y_label: &str,
    interval: f64,
    mapping: &SampleMapping,
) -> Result<(), String> {
    if panels.is_empty() {
        return Err("no panels to plot".to_string());
    }
    let mut all_y = Vec::new();
    for panel in panels {
        all_y.extend_from_slice(&panel.y_values);
    }
    let shared_ylim = percentile_ylim(&all_y);
    let summaries: Vec<Option<SampleSummary>> = panels
        .iter()
        .map(|panel| sample_summary_curves(&panel.traces, interval))
        .collect();
    let mut summary_values = Vec::new();
    for summary in summaries.iter().flatten() {
        summary_values.extend_from_slice(&summary.mean);
        summary_values.extend_from_slice(&summary.median);
        summary_values.extend_from_slice(&summary.q25);
        summary_values.extend_from_slice(&summary.q75);
    }
    let shared_summary = percentile_ylim(&summary_values);

    for (index, panel) in panels.iter().enumerate() {
        let dir = sample_dir(panel.sample);
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let primary = dir.join(format!("{file_stem}.png"));
        let own = percentile_ylim(&panel.y_values);
        let one = std::slice::from_ref(panel);
        let summary = summaries.get(index).unwrap_or(&None);
        let own_summary = summary_ylim(summary.as_ref());
        write_subplot_grid(one, &primary, y_label, interval, None, mapping, |_| own)?;
        write_subplot_grid(
            one,
            &companion_plot_path(&primary, "shared_y"),
            y_label,
            interval,
            None,
            mapping,
            |_| shared_ylim,
        )?;
        let summary_plot = companion_plot_path(&primary, "summary");
        write_summary_subplot_grid(
            one,
            std::slice::from_ref(summary),
            &summary_plot,
            y_label,
            None,
            mapping,
            |_| own_summary,
        )?;
        write_summary_subplot_grid(
            one,
            std::slice::from_ref(summary),
            &companion_plot_path(&summary_plot, "shared_y"),
            y_label,
            None,
            mapping,
            |_| shared_summary,
        )?;
    }
    Ok(())
}

fn write_subplot_grid(
    panels: &[TracePanel],
    output_plot: &Path,
    y_label: &str,
    interval: f64,
    columns: Option<usize>,
    mapping: &SampleMapping,
    ylim_for_panel: impl Fn(usize) -> (f64, f64),
) -> Result<(), String> {
    let (rows, cols) = resolve_subplot_grid(panels.len(), columns);
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, panel) in panels.iter().enumerate() {
        let (y_low, y_high) = ylim_for_panel(index);
        let max_t = panel
            .traces
            .iter()
            .flat_map(|trace| trace.iter().map(|point| point.0))
            .fold(0.0f64, f64::max)
            * interval;
        let (color, alpha) = trace_color_alpha(&sample_trace_naming_haystack(
            panel.sample,
            &panel.paths,
            mapping,
        ));
        let title = sample_subplot_title(panel.sample, panel.traces.len(), mapping);
        let traces = panel.traces.clone();
        let y_label = y_label.to_string();
        let y_scientific = y_label.contains("intensity");

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |panel| {
            for trace in &traces {
                let x: Vec<f64> = trace.iter().map(|(time, _)| time * interval).collect();
                let y: Vec<f64> = trace.iter().map(|(_, value)| *value).collect();
                panel.line(&x, &y, trace_line_style(color, alpha));
            }
            let mut axes = AxesStyle::new()
                .title(title)
                .x_label("time (min)")
                .y_label(y_label)
                .y_range(y_low, y_high)
                .x_range(0.0, max_t.max(interval));
            if y_scientific {
                axes = axes.y_tick_format(TickFormat::Scientific);
            }
            panel.axes(axes);
        });
    }

    for index in panels.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    let pad = if panels.len() <= 1 {
        SAVE_PAD_SINGLE_INCHES
    } else {
        SAVE_PAD_GRID_INCHES
    };
    save_figure(&figure, output_plot, pad)
}

fn write_summary_subplot_grid(
    panels: &[TracePanel],
    summaries: &[Option<SampleSummary>],
    output_plot: &Path,
    y_label: &str,
    columns: Option<usize>,
    mapping: &SampleMapping,
    ylim_for_panel: impl Fn(usize) -> (f64, f64),
) -> Result<(), String> {
    let (rows, cols) = resolve_subplot_grid(panels.len(), columns);
    let mut builder = figure_builder_for_grid(rows, cols);

    for (index, panel) in panels.iter().enumerate() {
        let (y_low, y_high) = ylim_for_panel(index);
        let (color, _alpha) = trace_color_alpha(&sample_trace_naming_haystack(
            panel.sample,
            &panel.paths,
            mapping,
        ));
        let y_label = y_label.to_string();
        let y_scientific = y_label.contains("intensity");
        let summary = summaries.get(index).and_then(|value| value.as_ref());
        let title = sample_subplot_title(
            panel.sample,
            summary.map(|s| s.trace_count).unwrap_or(0),
            mapping,
        );
        let show_legend = index == 0;

        builder = builder.panel(GridPos::new(rows, cols, index + 1), move |p| {
            let max_t = if let Some(summary) = summary {
                p.fill_between(
                    &summary.t_minutes,
                    &summary.q25,
                    &summary.q75,
                    FillBetweenStyle::new()
                        .color(Color::hex(color))
                        .alpha(0.25)
                        .label("IQR"),
                );
                p.line(
                    &summary.t_minutes,
                    &summary.median,
                    trace_line_style(color, 1.0)
                        .width(1.8)
                        .dash(LineDash::Solid)
                        .label("median"),
                );
                p.line(
                    &summary.t_minutes,
                    &summary.mean,
                    trace_line_style(color, 1.0)
                        .width(1.5)
                        .dash(LineDash::Dashed)
                        .label("mean"),
                );
                summary
                    .t_minutes
                    .iter()
                    .copied()
                    .fold(0.0f64, f64::max)
                    .max(1.0)
            } else {
                1.0
            };

            let mut axes = AxesStyle::new()
                .title(title)
                .x_label("time (min)")
                .y_label(y_label)
                .y_range(y_low, y_high)
                .x_range(0.0, max_t);
            if y_scientific {
                axes = axes.y_tick_format(TickFormat::Scientific);
            }
            if show_legend && summary.is_some() {
                axes = axes.legend(LegendStyle::show());
            }
            p.axes(axes);
        });
    }

    for index in panels.len()..(rows * cols) {
        builder = builder.panel(GridPos::new(rows, cols, index + 1), |panel| {
            panel.axes(AxesStyle::new().hide(true));
        });
    }

    let figure = builder.build().map_err(|error| error.to_string())?;
    let pad = if panels.len() <= 1 {
        SAVE_PAD_SINGLE_INCHES
    } else {
        SAVE_PAD_GRID_INCHES
    };
    save_figure(&figure, output_plot, pad)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::engagement::write_engagement_summary;
    use crate::sample::SampleAnalysis;

    #[test]
    fn subplot_grid_shape_one_to_twelve() {
        assert_eq!(subplot_grid_shape(1), (1, 1));
        assert_eq!(subplot_grid_shape(2), (1, 2));
        assert_eq!(subplot_grid_shape(3), (2, 2));
        assert_eq!(subplot_grid_shape(4), (2, 2));
        assert_eq!(subplot_grid_shape(5), (2, 3));
        assert_eq!(subplot_grid_shape(6), (2, 3));
        assert_eq!(subplot_grid_shape(7), (3, 3));
        assert_eq!(subplot_grid_shape(9), (3, 3));
        assert_eq!(subplot_grid_shape(10), (3, 4));
        assert_eq!(subplot_grid_shape(12), (3, 4));
    }

    #[test]
    fn engagement_plots_count_spots_without_replacing_death_reporter_figures() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("analysis/Pos41")).unwrap();
        fs::write(
            root.join("analysis/Pos41/engagement.csv"),
            "\
roi,t,minutes,tumor_cells,t_cells,engagements
0,0,0.0000,1,0,0
0,5,13.7500,1,3,1
1,0,0.0000,1,1,0
1,5,13.7500,1,2,0
",
        )
        .unwrap();
        let mapping = SampleMapping(vec![SampleAnalysis {
            name: "CMRA_0_1".into(),
            positions: vec![41],
            signal: vec![1],
            segmentation: 0,
        }]);
        run_plot_engagement_traces(root, &mapping, 2.75).unwrap();
        write_engagement_summary(root, &mapping).unwrap();
        assert!(root
            .join("results/CMRA_0_1/engagement_traces.png")
            .is_file());
        assert!(root
            .join("results/CMRA_0_1/engagement_traces_summary_shared_y.png")
            .is_file());
        assert!(root.join("results/CMRA_0_1/engagement.xlsx").is_file());
        assert!(root
            .join("results/CMRA_0_1/engagement_summary.xlsx")
            .is_file());
        assert!(root.join("analysis/Pos41/engagement_summary.csv").is_file());
        assert!(!root.join("results/traces.png").exists());
        assert!(!root.join("results/engagement_traces.png").exists());
    }

    #[test]
    fn death_reporter_pipeline_writes_survival_kill_and_trace_figures_without_model() {
        use tiff::encoder::{colortype, TiffEncoder};

        fn write_pages(path: &Path, pages: &[Vec<u8>]) {
            let file = fs::File::create(path).expect("create tiff");
            let mut encoder = TiffEncoder::new(file).expect("encoder");
            for page in pages {
                let image = encoder
                    .new_image::<colortype::Gray8>(2, 2)
                    .expect("gray image");
                image.write_data(page).expect("write page");
            }
        }

        fn write_position(workspace: &Path, position: u32, signal_frames: &[[u8; 4]]) {
            let pos_dir = workspace.join(format!("roi/Pos{position}"));
            fs::create_dir_all(&pos_dir).unwrap();
            let mut pages = Vec::with_capacity(signal_frames.len() * 2);
            for frame in signal_frames {
                pages.push(vec![255; 4]);
                pages.push(frame.to_vec());
            }
            write_pages(&pos_dir.join("Roi0.tif"), &pages);
            let index = serde_json::json!({
                "position": position,
                "axisOrder": "TCZYX",
                "channelCount": 2,
                "timeCount": signal_frames.len(),
                "zCount": 1,
                "rois": [{
                    "roi": 0,
                    "fileName": "Roi0.tif",
                    "bbox": { "roi": 0, "x": 0, "y": 0, "w": 2, "h": 2 }
                }]
            });
            fs::write(
                pos_dir.join("index.json"),
                serde_json::to_string(&index).unwrap(),
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write_position(
            root,
            1,
            &[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 5]],
        );
        write_position(
            root,
            2,
            &[
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [200, 210, 220, 230],
                [205, 215, 225, 235],
            ],
        );
        let mapping = SampleMapping(vec![
            SampleAnalysis {
                name: "low".into(),
                positions: vec![1],
                signal: vec![1],
                segmentation: 0,
            },
            SampleAnalysis {
                name: "high".into(),
                positions: vec![2],
                signal: vec![1],
                segmentation: 0,
            },
        ]);

        crate::run_death_reporter_pipeline(root, &mapping, 2.75).unwrap();

        assert!(root.join("analysis/Pos1/ch1.csv").is_file());
        assert!(root.join("analysis/Pos2/ch1.csv").is_file());
        assert!(root.join("results/low/traces.png").is_file());
        assert!(root.join("results/low/traces_summary.png").is_file());
        assert!(root
            .join("results/low/traces_summary_shared_y.png")
            .is_file());
        assert!(root.join("results/high/traces.png").is_file());
        assert!(root.join("results/low/traces.xlsx").is_file());
        assert!(root.join("results/high/traces.xlsx").is_file());
        assert!(root.join("results/kill_curve.csv").is_file());
        assert!(root.join("results/survival_curve.png").is_file());
        assert!(root.join("results/kill_curve.png").is_file());
        assert!(root.join("results/death_times.csv").is_file());
        assert!(root.join("results/death_times.png").is_file());
    }
}

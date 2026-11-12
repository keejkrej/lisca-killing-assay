//! Workspace trace CSV loading (`analysis/Pos{n}/ch{m}.csv`) and per-sample
//! `results/<sample>/traces.xlsx` publication.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::csv_io::{column_index, parse_f64, read_csv};
use crate::export::write_xlsx_only;
use crate::sample::{sample_pack_dirnames, SampleMapping};

const TRACE_HEADERS: [&str; 7] = ["pos", "roi", "t", "area", "background", "sum", "corrected"];
const TRACE_HEADERS_WITH_CHANNEL: [&str; 8] = [
    "pos",
    "channel",
    "roi",
    "t",
    "area",
    "background",
    "sum",
    "corrected",
];

#[derive(Debug, Clone)]
pub(crate) struct TracePanel {
    pub sample: usize,
    pub paths: Vec<PathBuf>,
    pub traces: Vec<Vec<(f64, f64)>>,
    pub y_values: Vec<f64>,
}

pub(crate) type TracePointGroup = BTreeMap<i64, Vec<(f64, f64)>>;

/// Discover `<dir>/Pos{n}/ch{n}.csv` files.
pub(crate) fn discover_trace_csvs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Err(format!("Expected trace directory at {}", dir.display()));
    }
    let mut csvs = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let pos_dir = entry.path();
        if !pos_dir.is_dir() {
            continue;
        }
        let Some(pos_name) = pos_dir.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !pos_name.starts_with("Pos") {
            continue;
        }
        for child in std::fs::read_dir(&pos_dir).map_err(|error| error.to_string())? {
            let child = child.map_err(|error| error.to_string())?;
            let path = child.path();
            if path.extension().is_some_and(|extension| extension == "csv")
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| {
                        stem.starts_with("ch") && stem[2..].chars().all(|c| c.is_ascii_digit())
                    })
            {
                csvs.push(path);
            }
        }
    }
    csvs.sort_by(|left, right| {
        (
            left.parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_owned()),
            left.file_name().map(|n| n.to_owned()),
        )
            .cmp(&(
                right
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_owned()),
                right.file_name().map(|n| n.to_owned()),
            ))
    });
    if csvs.is_empty() {
        return Err(format!(
            "No position metrics CSV files (expected Pos{{n}}/ch{{n}}.csv) in {}",
            dir.display()
        ));
    }
    Ok(csvs)
}

/// Parse `(position, signal_channel)` from `.../Pos{n}/ch{n}.csv`.
pub fn parse_trace_path(path: &Path) -> Result<(u32, u32), String> {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("invalid trace path {}", path.display()))?;
    let channel = stem
        .strip_prefix("ch")
        .and_then(|rest| rest.parse::<u32>().ok())
        .ok_or_else(|| {
            format!(
                "Expected trace path Pos{{n}}/ch{{n}}.csv, got {}",
                path.display()
            )
        })?;
    let parent = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Expected trace path Pos{{n}}/ch{{n}}.csv, got {}",
                path.display()
            )
        })?;
    let position = parent
        .strip_prefix("Pos")
        .and_then(|rest| rest.parse::<u32>().ok())
        .ok_or_else(|| {
            format!(
                "Expected trace path Pos{{n}}/ch{{n}}.csv, got {}",
                path.display()
            )
        })?;
    Ok((position, channel))
}

/// Resolve the Sample (index in assay order) that owns a trace CSV path.
pub fn resolve_sample(path: &Path, mapping: &SampleMapping) -> Result<usize, String> {
    let (position, signal_channel) = parse_trace_path(path)?;
    let mut matches = mapping.iter().enumerate().filter_map(|(index, sample)| {
        (sample.signal.contains(&signal_channel) && sample.positions.contains(&position))
            .then_some(index)
    });
    let Some(index) = matches.next() else {
        return Err(format!(
            "No assay mapping entry for Pos{position} signal channel {signal_channel} ({})",
            path.display()
        ));
    };
    if let Some(other) = matches.next() {
        let name = |idx: usize| mapping.get(idx).map(|sample| sample.name.clone());
        return Err(format!(
            "Ambiguous sample for Pos{position} signal channel {signal_channel}: {:?} and {:?}",
            name(index).unwrap_or_default(),
            name(other).unwrap_or_default()
        ));
    }
    Ok(index)
}

pub(crate) fn load_trace_panel(path: &Path, y_column: &str) -> Result<TracePanel, String> {
    let (headers, rows) = read_csv(path)?;
    let groups = group_trace_rows(&headers, &rows, y_column)?;
    let mut y_values = Vec::new();
    let traces = groups
        .into_values()
        .map(|mut points| {
            points.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            y_values.extend(points.iter().map(|(_, value)| *value));
            points
        })
        .collect();
    Ok(TracePanel {
        sample: 0,
        paths: vec![path.to_path_buf()],
        traces,
        y_values,
    })
}

pub(crate) fn load_trace_panels_by_sample(
    csvs: &[PathBuf],
    y_column: &str,
    mapping: &SampleMapping,
) -> Result<Vec<TracePanel>, String> {
    let mut grouped: BTreeMap<usize, TracePanel> = BTreeMap::new();
    for path in csvs {
        let sample = resolve_sample(path, mapping)?;
        let panel = load_trace_panel(path, y_column)?;
        let entry = grouped.entry(sample).or_insert_with(|| TracePanel {
            sample,
            paths: Vec::new(),
            traces: Vec::new(),
            y_values: Vec::new(),
        });
        entry.paths.extend(panel.paths);
        entry.traces.extend(panel.traces);
        entry.y_values.extend(panel.y_values);
    }
    Ok(grouped.into_values().collect())
}

pub(crate) fn group_trace_rows(
    headers: &[String],
    rows: &[Vec<String>],
    y_column: &str,
) -> Result<TracePointGroup, String> {
    let t_index = column_index(headers, "t").ok_or("missing t column")?;
    let y_index =
        column_index(headers, y_column).ok_or_else(|| format!("missing {y_column} column"))?;
    let roi_index = column_index(headers, "roi").ok_or("missing roi column")?;

    let mut groups: TracePointGroup = BTreeMap::new();
    for row in rows {
        let roi = parse_f64(&row[roi_index]).ok_or("invalid roi")? as i64;
        let t = parse_f64(&row[t_index]).ok_or("invalid t")?;
        let y = parse_f64(&row[y_index]).ok_or("invalid y")?;
        groups.entry(roi).or_default().push((t, y));
    }
    Ok(groups)
}

/// Write per-sample `results/<sample>/traces.xlsx` from `analysis/Pos{n}/ch{m}.csv`.
pub fn publish_sample_traces_xlsx(
    workspace: &Path,
    mapping: &SampleMapping,
) -> Result<Vec<PathBuf>, String> {
    if mapping.is_empty() {
        return Err("samples[] is empty".to_string());
    }
    let csvs = discover_trace_csvs(&workspace.join("analysis"))?;
    let dirnames = sample_pack_dirnames(mapping);
    let mut frames: BTreeMap<usize, Vec<Vec<String>>> = BTreeMap::new();
    let multi_channel = mapping.iter().any(|entry| entry.signal.len() > 1);
    for csv_path in csvs {
        let Ok(sample) = resolve_sample(&csv_path, mapping) else {
            continue;
        };
        let (position, signal) = parse_trace_path(&csv_path)?;
        let Some(entry) = mapping.get(sample) else {
            continue;
        };
        if !entry.signal.contains(&signal) {
            continue;
        }
        let (headers, rows) = read_csv(&csv_path)?;
        let roi_index = column_index(&headers, "roi").ok_or("missing roi")?;
        let t_index = column_index(&headers, "t").ok_or("missing t")?;
        let area_index = column_index(&headers, "area").ok_or("missing area")?;
        let background_index = column_index(&headers, "background").ok_or("missing background")?;
        let sum_index = column_index(&headers, "sum").ok_or("missing sum")?;
        let corrected_index = column_index(&headers, "corrected").ok_or("missing corrected")?;
        let pos_index = column_index(&headers, "pos");
        for row in rows {
            let pos = pos_index
                .and_then(|index| parse_f64(&row[index]).map(|value| value as i64))
                .unwrap_or(i64::from(position));
            let mut out = vec![pos.to_string()];
            if multi_channel {
                out.push(signal.to_string());
            }
            out.extend([
                row[roi_index].clone(),
                row[t_index].clone(),
                row[area_index].clone(),
                row[background_index].clone(),
                row[sum_index].clone(),
                row[corrected_index].clone(),
            ]);
            frames.entry(sample).or_default().push(out);
        }
    }

    let headers: &[&str] = if multi_channel {
        &TRACE_HEADERS_WITH_CHANNEL
    } else {
        &TRACE_HEADERS
    };
    let mut written = Vec::new();
    for (sample, mut rows) in frames {
        let Some(dirname) = dirnames.get(&sample) else {
            continue;
        };
        rows.sort_by(|left, right| {
            let pos = |row: &[String]| row[0].parse::<i64>().unwrap_or(0);
            let channel_or_roi = |row: &[String]| row[1].parse::<i64>().unwrap_or(0);
            let roi_or_t = |row: &[String]| row[2].parse::<i64>().unwrap_or(0);
            let t = |row: &[String]| {
                if multi_channel {
                    row[3].parse::<i64>().unwrap_or(0)
                } else {
                    0
                }
            };
            pos(left)
                .cmp(&pos(right))
                .then(channel_or_roi(left).cmp(&channel_or_roi(right)))
                .then(roi_or_t(left).cmp(&roi_or_t(right)))
                .then(t(left).cmp(&t(right)))
        });
        let output = workspace.join("results").join(dirname).join("traces.xlsx");
        write_xlsx_only(&output, headers, &rows)?;
        written.push(output);
    }
    if written.is_empty() {
        return Err("No analysis traces matched samples[]".to_string());
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::SampleAnalysis;

    #[test]
    fn parse_trace_path_reads_pos_and_channel() {
        let path = Path::new("/ws/traces/Pos7/ch2.csv");
        assert_eq!(parse_trace_path(path).unwrap(), (7, 2));
    }

    #[test]
    fn resolve_sample_uses_mapping() {
        let mapping = SampleMapping(vec![
            SampleAnalysis {
                name: "A".into(),
                positions: vec![1],
                signal: vec![2],
                segmentation: 0,
            },
            SampleAnalysis {
                name: "B".into(),
                positions: vec![7],
                signal: vec![2],
                segmentation: 0,
            },
        ]);
        let path = Path::new("/ws/traces/Pos7/ch2.csv");
        assert_eq!(resolve_sample(path, &mapping).unwrap(), 1);
    }
}

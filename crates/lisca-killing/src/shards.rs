//! Sharded classifier output merge (`traces/Pos{N}/ch{M}.csv` + `results/predictions.csv`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Merge per-position classifier prediction shards into `workspace` with CSV
/// header deduplication. Binary sidecars (`.xlsx`) are skipped.
pub fn merge_prediction_shards(workspace: &Path, shards: &[PathBuf]) -> Result<(), String> {
    let mut files = BTreeMap::<PathBuf, Vec<PathBuf>>::new();
    for shard in shards {
        for relative_dir in ["traces", "results"] {
            let directory = shard.join(relative_dir);
            collect_shard_files(&directory, shard, &mut files)?;
        }
    }
    for (relative, parts) in files {
        let target = workspace.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut merged = String::new();
        for (index, part) in parts.iter().enumerate() {
            let contents = fs::read_to_string(part)
                .map_err(|error| format!("failed to read {}: {error}", part.display()))?;
            let mut lines = contents.lines();
            let header = lines
                .next()
                .ok_or_else(|| format!("prediction shard is empty: {}", part.display()))?;
            if index == 0 {
                merged.push_str(header);
                merged.push('\n');
            }
            for line in lines {
                if !line.trim().is_empty() {
                    merged.push_str(line);
                    merged.push('\n');
                }
            }
        }
        let staging = target.with_extension("analysis-stage");
        fs::write(&staging, merged).map_err(|error| error.to_string())?;
        fs::rename(&staging, &target).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn collect_shard_files(
    directory: &Path,
    shard: &Path,
    files: &mut BTreeMap<PathBuf, Vec<PathBuf>>,
) -> Result<(), String> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("failed to list {}: {error}", directory.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read an entry in {}: {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_shard_files(&path, shard, files)?;
        } else if path.is_file() && path.extension().is_some_and(|ext| ext == "csv") {
            let relative = path
                .strip_prefix(shard)
                .map_err(|error| error.to_string())?;
            files.entry(relative.to_path_buf()).or_default().push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::run_plot_traces;
    use crate::sample::{SampleAnalysis, SampleMapping};

    fn unique_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "lisca-killing-merge-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn prediction_shards_merge_without_repeating_headers() {
        let root = unique_root("basic");
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        let first = root.join("first");
        let second = root.join("second");
        for shard in [&first, &second] {
            fs::create_dir_all(shard.join("results")).unwrap();
        }
        fs::write(
            first.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.1,true,0,A\n",
        )
        .unwrap();
        fs::write(
            second.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.9,false,1,A\n",
        )
        .unwrap();

        merge_prediction_shards(&workspace, &[first, second]).unwrap();
        let merged = fs::read_to_string(workspace.join("results/predictions.csv")).unwrap();
        assert_eq!(
            merged
                .lines()
                .filter(|line| line.starts_with("t,crop"))
                .count(),
            1
        );
        assert!(merged.contains("true,0,A"));
        assert!(merged.contains("false,1,A"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merge_prediction_shards_preserves_nested_trace_csvs() {
        let root = unique_root("nested");
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        let shard = root.join("sc0-Pos1");
        fs::create_dir_all(shard.join("traces/Pos1")).unwrap();
        fs::create_dir_all(shard.join("results")).unwrap();
        fs::write(
            shard.join("traces/Pos1/ch0.csv"),
            "roi,t,p_dead\n0,0,0.1\n0,1,0.2\n",
        )
        .unwrap();
        fs::write(
            shard.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.1,false,1,A\n",
        )
        .unwrap();

        merge_prediction_shards(&workspace, &[shard]).unwrap();

        assert!(workspace.join("results/predictions.csv").is_file());
        let ts = workspace.join("traces/Pos1/ch0.csv");
        assert!(
            ts.is_file(),
            "nested trace CSV was not merged into the workspace"
        );
        assert_eq!(
            fs::read_to_string(&ts).unwrap(),
            "roi,t,p_dead\n0,0,0.1\n0,1,0.2\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merge_prediction_shards_merges_disjoint_positions_across_shards() {
        let root = unique_root("disjoint");
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        let shard1 = root.join("sc0-Pos1");
        let shard2 = root.join("sc0-Pos2");
        fs::create_dir_all(shard1.join("traces/Pos1")).unwrap();
        fs::create_dir_all(shard1.join("results")).unwrap();
        fs::write(
            shard1.join("traces/Pos1/ch0.csv"),
            "roi,t,p_dead\n0,0,0.1\n",
        )
        .unwrap();
        fs::write(
            shard1.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.1,false,1,A\n",
        )
        .unwrap();
        fs::create_dir_all(shard2.join("traces/Pos2")).unwrap();
        fs::create_dir_all(shard2.join("results")).unwrap();
        fs::write(
            shard2.join("traces/Pos2/ch0.csv"),
            "roi,t,p_dead\n0,0,0.9\n",
        )
        .unwrap();
        fs::write(
            shard2.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.9,true,2,A\n",
        )
        .unwrap();

        merge_prediction_shards(&workspace, &[shard1, shard2]).unwrap();

        assert_eq!(
            fs::read_to_string(workspace.join("traces/Pos1/ch0.csv")).unwrap(),
            "roi,t,p_dead\n0,0,0.1\n"
        );
        assert_eq!(
            fs::read_to_string(workspace.join("traces/Pos2/ch0.csv")).unwrap(),
            "roi,t,p_dead\n0,0,0.9\n"
        );
        let merged = fs::read_to_string(workspace.join("results/predictions.csv")).unwrap();
        assert_eq!(
            merged
                .lines()
                .filter(|line| line.starts_with("t,crop"))
                .count(),
            1
        );
        assert!(merged.contains("false,1,A"));
        assert!(merged.contains("true,2,A"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merge_prediction_shards_skips_missing_traces_directory() {
        let root = unique_root("missing-ts");
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        let shard = root.join("sc0-Pos1");
        fs::create_dir_all(shard.join("results")).unwrap();
        fs::write(
            shard.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.1,false,1,A\n",
        )
        .unwrap();

        merge_prediction_shards(&workspace, &[shard]).unwrap();

        assert!(workspace.join("results/predictions.csv").is_file());
        assert!(!workspace.join("traces").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merge_skips_binary_xlsx_sidecars_and_renders_traces() {
        let root = unique_root("xlsx");
        let _ = fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        let shard = root.join("sc0-Pos1");
        fs::create_dir_all(shard.join("traces/Pos1")).unwrap();
        fs::create_dir_all(shard.join("results")).unwrap();

        write_csv_with_binary_sidecar(
            &shard.join("results/predictions.csv"),
            "t,crop,p_dead,label,pos,sample\n0,1,0.1,false,1,A\n1,1,0.3,false,1,A\n",
        );
        write_csv_with_binary_sidecar(
            &shard.join("traces/Pos1/ch0.csv"),
            "roi,t,area,background,sum,corrected\n0,0,4,1,10,6\n0,1,4,1,12,8\n",
        );

        merge_prediction_shards(&workspace, std::slice::from_ref(&shard)).unwrap();

        assert!(workspace.join("results/predictions.csv").is_file());
        assert!(!workspace.join("results/predictions.xlsx").exists());
        assert!(workspace.join("traces/Pos1/ch0.csv").is_file());
        assert!(!workspace.join("traces/Pos1/ch0.xlsx").exists());

        let mapping = SampleMapping(vec![SampleAnalysis {
            name: "A".into(),
            positions: vec![1],
            signal: vec![0],
            segmentation: 0,
        }]);

        fs::create_dir_all(workspace.join("analysis/Pos1")).unwrap();
        fs::copy(
            workspace.join("traces/Pos1/ch0.csv"),
            workspace.join("analysis/Pos1/ch0.csv"),
        )
        .unwrap();
        run_plot_traces(&workspace, &mapping, 30.0, None).unwrap();
        assert!(workspace.join("results/A/traces.png").is_file());
        assert!(workspace.join("results/A/traces.xlsx").is_file());
        assert!(!workspace.join("results/traces.png").exists());
        fs::remove_dir_all(root).unwrap();
    }

    fn write_csv_with_binary_sidecar(path: &Path, csv: &str) {
        fs::write(path, csv).unwrap();
        fs::write(path.with_extension("xlsx"), [0xff, 0xfe, 0x00]).unwrap();
    }
}

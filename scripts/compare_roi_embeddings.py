"""Sample ROI TIFFs, embed on a GPU host, and compare kNN with a linear SVM."""

import argparse
import json
from pathlib import Path

from apoptosis.embedding.experiment import (
    compare_sample,
    embed_sample,
    import_frame_manifest,
    review_sequences,
    sample_rois,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    labeled = commands.add_parser("import-frames")
    labeled.add_argument("manifest_path", type=Path)
    labeled.add_argument("output", type=Path)
    labeled.add_argument("--workspace", type=Path)
    labeled.add_argument("--frame-stride", type=int, default=10)
    labeled.add_argument("--label-map", type=json.loads)
    labeled.add_argument("--label-source", default="existing-frame-manifest")
    sample = commands.add_parser("sample")
    sample.add_argument("workspace", type=Path)
    sample.add_argument("output", type=Path)
    sample.add_argument("--roi-count", type=int, default=24)
    sample.add_argument("--frames-per-roi", type=int, default=3)
    sample.add_argument("--channel", type=int, default=0)
    sample.add_argument("--z", type=int, default=0)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--contrast", choices=["baseline", "frame"], default="baseline")
    sequences = commands.add_parser("sequences")
    sequences.add_argument("directory", type=Path)
    sequences.add_argument("--frames", type=int, default=8)
    embed = commands.add_parser("embed")
    embed.add_argument("directory", type=Path)
    embed.add_argument("--model-id", default="google/embeddinggemma-2")
    embed.add_argument("--revision")
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--batch-size", type=int, default=8)
    compare = commands.add_parser("compare")
    compare.add_argument("directory", type=Path)
    compare.add_argument("--neighbors", type=int, default=3)
    compare.add_argument("--svm-c", type=float, default=1.0)
    options = vars(parser.parse_args())
    command = options.pop("command")
    try:
        if command == "import-frames":
            result = import_frame_manifest(**options)
            print(f"Imported {len(result['rows'])} labeled images")
        elif command == "sample":
            result = sample_rois(**options)
            print(f"Sampled {len(result['rows'])} images")
        elif command == "sequences":
            result = review_sequences(**options)
            print(f"Rendered {len(result['groups'])} ROI sequences")
        elif command == "embed":
            result = embed_sample(**options)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ("model_id", "revision", "shape", "cache_reused")
                    },
                    indent=2,
                )
            )
        else:
            result = compare_sample(**options)
            print(json.dumps(result["metrics"], indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()

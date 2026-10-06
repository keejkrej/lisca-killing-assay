"""Measure warm GPU embedding throughput and CPU classifier costs separately."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    metadata = json.loads((root / "embeddings.json").read_text())
    rows = json.loads((root / "manifest.json").read_text())["rows"]
    selected = np.random.default_rng(42).choice(len(rows), 32, replace=False)
    start = time.perf_counter()
    images = []
    for i in selected:
        with Image.open(root / rows[i]["image"]) as im:
            images.append({"image": im.convert("RGB")})
    decode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = SentenceTransformer(
        metadata["model_id"],
        revision=metadata["revision"],
        device="cuda",
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    report = {
        "gpu": torch.cuda.get_device_name(),
        "host": platform.node(),
        "torch": torch.__version__,
        "model_id": metadata["model_id"],
        "revision": metadata["revision"],
        "dtype": "float32",
        "image_count_per_repeat": len(images),
        "repeats": 3,
        "png_decode_seconds": decode_seconds,
        "model_load_seconds": load_seconds,
        "note": "Warm encode includes processor, transfers, model and numpy output. "
        "Excludes TIFF crop/preprocessing, network, queueing and disk embedding cache. "
        "32 existing RGB PNGs sampled with seed 42; each batch size warmed separately.",
        "embedding": [],
    }
    with torch.inference_mode():
        for batch_size in (1, 8, 16, 32):
            try:
                model.encode(
                    images[:batch_size],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model.encode(
                        images,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - start)
                median = float(np.median(elapsed))
                result = {
                    "batch_size": batch_size,
                    "seconds": elapsed,
                    "images_per_second": len(images) / median,
                    "amortized_ms_per_image": median * 1000 / len(images),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            except torch.cuda.OutOfMemoryError:
                result = {"batch_size": batch_size, "error": "CUDA out of memory"}
                torch.cuda.empty_cache()
            report["embedding"].append(result)
            print(json.dumps(result), flush=True)
            (root / "speed-benchmark.json").write_text(json.dumps(report, indent=2))

    vectors = np.load(root / "embeddings.npy").astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ref = np.array([i for i, r in enumerate(rows) if r["split"] == "reference"])
    query = np.array([i for i, r in enumerate(rows) if r["split"] == "evaluation"])
    labels = np.array([rows[i]["label"] for i in ref])
    references, queries = vectors[ref], vectors[query]
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        svm = LinearSVC(C=1, class_weight="balanced", random_state=0, max_iter=10000)
        svm.fit(references, labels)
        report["svm_fit_seconds"] = time.perf_counter() - start

        def knn():
            similarities = queries @ references.T
            indices = np.argsort(-similarities, axis=1, kind="stable")[:, :3]
            weights = 1 / np.maximum(
                1 - np.take_along_axis(similarities, indices, axis=1), 1e-6
            )
            dead = (weights * (labels[indices] == "dead")).sum(axis=1)
            return dead >= weights.sum(axis=1) / 2

        for name, operation in (("knn", knn), ("svm", lambda: svm.predict(queries))):
            operation()
            elapsed = []
            for _ in range(20):
                start = time.perf_counter()
                operation()
                elapsed.append(time.perf_counter() - start)
            report[name] = {
                "queries": len(query),
                "references": len(ref),
                "cpu_blas_threads": 1,
                "median_ms_per_query": float(np.median(elapsed)) * 1000 / len(query),
            }
    (root / "speed-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Export the frozen UPC155 v3 checkpoints for browser inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def export(project, output):
    source = project / "experiments" / "main_v3_all"
    weights = bytearray()
    seeds = []
    tensors = None
    arch = None
    for seed in (11, 42, 155):
        run = source / f"seed_{seed}"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        checkpoint = run / "model.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != config["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")

        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if bundle["features"] != "phys_wiener_22d" or bundle["architecture"] != config["architecture"]:
            raise ValueError(f"Unexpected checkpoint architecture for seed {seed}")
        start = len(weights)
        layout = []
        for name, tensor in bundle["state"].items():
            array = tensor.detach().numpy().astype("<f4", copy=False)
            layout.append({"name": name, "offset": len(weights) - start, "count": array.size})
            weights.extend(array.tobytes())

        architecture = {
            "inputs": bundle["stats"]["xm"].size,
            "hidden": config["architecture"]["hidden_dim"],
            "depth": config["architecture"]["depth"],
            "components": bundle["k"],
            "parameters": (len(weights) - start) // 4,
        }
        if architecture["parameters"] != config["parameters"] or bundle["k"] != config["k"]:
            raise ValueError(f"Unexpected parameter count for seed {seed}")
        if tensors is None:
            tensors, arch = layout, architecture
        elif layout != tensors or architecture != arch:
            raise ValueError(f"Tensor layout mismatch for seed {seed}")

        stats = {key: np.asarray(bundle["stats"][key]).ravel().tolist() for key in ("xm", "xs")}
        seeds.append({
            "seed": seed,
            "checkpointSha256": digest,
            "byteOffset": start,
            "stats": stats,
            "calibration": config["calibration"],
        })

    meta = {
        "version": "wiener-v3",
        "weightsFile": "weights.bin",
        "weightsBytes": len(weights),
        "weightsSha256": hashlib.sha256(weights).hexdigest(),
        "arch": arch,
        "tensors": tensors,
        "seeds": seeds,
    }
    (output / meta["weightsFile"]).write_bytes(weights)
    (output / "model.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upc155", type=Path, help="Path to the UPC155 repository")
    args = parser.parse_args()
    export(args.upc155, Path(__file__).resolve().parent)

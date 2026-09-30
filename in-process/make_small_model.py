"""Cut gpt-oss-20b down to its first N layers (real weights), for a fast-loading repro model.

usage: python make_small_model.py <src dir> <dst dir> <n layers>
Copies tokenizer / template files, rewrites config.json (num_hidden_layers, layer_types),
writes one model.safetensors with embed, layers 0..N-1, final norm and lm_head.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def main(src, dst, n):
	os.makedirs(dst, exist_ok=True)
	for f in os.listdir(src):
		if f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json") or f.endswith(".jinja"):
			shutil.copy(os.path.join(src, f), dst)
	cfg = json.load(open(os.path.join(src, "config.json")))
	cfg["num_hidden_layers"] = n
	cfg["layer_types"] = cfg["layer_types"][:n]
	json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

	wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
	keep = [k for k in wmap if "layers." not in k or int(k.split("layers.")[1].split(".")[0]) < n]
	tensors = {}
	for shard in sorted(set(wmap[k] for k in keep)):
		with safe_open(os.path.join(src, shard), "pt") as f:
			for k in keep:
				if wmap[k] == shard:
					tensors[k] = f.get_tensor(k)
	save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
	print("wrote %d tensors, %d layers, %.2f GB" % (len(tensors), n, sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9))


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

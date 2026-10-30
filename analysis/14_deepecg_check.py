"""Load check for the second model, DeepECG-SL fine-tuned on MHI LVEF (regression). No ECG is scored.

    python3 -I analysis/14_deepecg_check.py

- Architecture: EfficientNet1DV2 from external/deepecg/DeepECG_Docker/notebooks/EfficientNetv2.py (torch only),
  MBConv expansion factor 2, with the regression head of notebooks/sl_fine_tuning.ipynb (pool, flatten, Linear(640, 1)).
- Weights: heartwise/deepecg-sl_finetuned_LVEF_MHI, LVEF_MSE_SL.pt (state_dict), loaded with weights_only=True.
  Its pickle references only tensors, OrderedDict, codecs.encode, numpy dtype and numpy.core.multiarray.scalar;
  the last three are allow-listed by name.
Strict load + a forward pass on random input -> results/deepecg_load_check.json.
"""
import codecs
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
DEEPECG = PROJ / "external/deepecg"
ARCH = DEEPECG / "DeepECG_Docker/notebooks/EfficientNetv2.py"
WEIGHTS = DEEPECG / "weights/deepecg-sl_finetuned_LVEF_MHI/LVEF_MSE_SL.pt"
ALLOWED = ([(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, codecs.encode]
           + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")])


class RegressionHead(nn.Module):
    """notebooks/sl_fine_tuning.ipynb, NewClassifier(out_dim=1)."""

    def __init__(self, out_dim=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(640, out_dim)

    def forward(self, x):
        return self.fc1(self.flatten(self.pool(x)))


def load_model(device="cpu"):
    spec = importlib.util.spec_from_file_location("deepecg_effnet", ARCH)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    # MBConv expansion factor 2 (the file default is 6): inferred from the checkpoint tensor shapes; the strict
    # load below verifies the whole configuration.
    model = arch.EfficientNet1DV2(expansion_factors=[1, 2, 2, 2, 2, 2, 2])
    model.classifier = RegressionHead(1)
    with torch.serialization.safe_globals(ALLOWED):
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)  # {seed, iter, score, metrics, state_dict}
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def main():
    model, ckpt = load_model()
    with torch.no_grad():
        y = model(torch.randn(4, 12, 2500))
    out = {"weights": str(WEIGHTS.relative_to(PROJ)), "strict_load": True,
           "n_parameters": int(sum(p.numel() for p in model.parameters())),
           "checkpoint_keys": sorted(ckpt), "seed": ckpt.get("seed"), "iter": ckpt.get("iter"),
           "score": float(ckpt["score"]) if ckpt.get("score") is not None else None,
           "metrics": json.loads(json.dumps(ckpt.get("metrics"), default=float)),
           "input_shape": [4, 12, 2500], "output_shape": list(y.shape)}
    (PROJ / "results/deepecg_load_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

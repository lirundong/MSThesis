import json
import os
from collections import OrderedDict

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"


def build_array(meta):
  # returns: (num_records, (step, value))
  arrs = {}
  for key, val in meta.items():
    if type(val) is str and os.path.isfile(val):
      with open(val, "r", encoding="utf-8") as f:
        v = json.load(f)
        va = np.array(v)[:, 1:]  # (step, value)
    else:
      ref_va = next(iter(arrs.values()))
      va = np.zeros_like(ref_va)
      va[:, 0] = ref_va[:, 0]
      va[:, 1] = val
    arrs[key] = va
  return arrs

def draw_curves(arrs, fmts, labels):
  fig, ax_w = plt.subplots(figsize=(4, 3), dpi=300)
  ax_lr = ax_w.twinx()
  handels = []
  color_seq = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  for i, (name, arr) in enumerate(arrs.items()):
    assert arr.shape[1] == 2
    ax = ax_lr if "LR" in name else ax_w
    alpha = 0.7 if "LR" in name else 1.0
    x = arr[:, 0]
    y = arr[:, 1]
    fmt = fmts[name]
    h, = ax.plot(x, y, fmt, label=labels[name], color=color_seq[i], alpha=alpha)
    handels.append(h)
  ax_w.set_title("$w_{f, q}$ vs. Learning Rate")
  ax_lr.legend(handles=handels, loc="lower right")
  ax_w.set_xlabel("training steps")
  ax_w.set_ylabel("$w_{f, q}$")
  ax_w.grid(axis="y", linestyle="--")
  ax_lr.tick_params(axis="y", labelcolor=color_seq[0])
  ax_lr.set_ylabel("Learning rate", color=color_seq[0])
  fig.tight_layout()
  return fig


if __name__ == "__main__":
  meta = OrderedDict([
    ("LR", "run-2019-09-03_15_14_58_idq_e120_b64x32_qa_fix_seg_sw_detach_kl_w1.0-tag-train_LR.json"),
    ("w_q", "run-2019-09-03_15_14_58_idq_e120_b64x32_qa_fix_seg_sw_detach_kl_w1.0-tag-train_soft_w.json"),
    ("w_f", 1.0),
    # ("train_acc", "run-2019-09-03_15_14_58_idq_e120_b64x32_qa_fix_seg_sw_detach_kl_w1.0-tag-train_train_q_top1.json"),
  ])
  fmts = {
    "LR": '-',
    "w_q": '--',
    "w_f": '-.',
    "train_acc": '-.',
  }
  labels = {
    "LR": "LR",
    "w_q": "$w_q$",
    "w_f": "$w_f$",
  }
  
  arrs = build_array(meta)
  fig = draw_curves(arrs, fmts, labels)
  fig.savefig("schedule.pdf", format="pdf")

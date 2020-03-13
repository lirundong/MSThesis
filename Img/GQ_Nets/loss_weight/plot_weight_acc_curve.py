import json
import glob
import os
import re
from collections import OrderedDict, defaultdict

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"


def load_run_data():
  """
  Return (dict):
    soft_w (tag): <array of data>
  """
  records = OrderedDict()
  for json_file in sorted(glob.glob("./schedule_data/*.json")):
    json_name = os.path.split(json_file)[-1].replace(".json", "")
    tag = json_name.split("-tag-")[-1].replace("train_", "")
    is_resumed = "resumed" in json_name
    with open(json_file, "r", encoding="utf-8") as f:
      data = json.load(f)
      data_arr = np.array(data)
    if is_resumed:
      prev_data_arr = records[tag]
      data_arr = np.concatenate([prev_data_arr, data_arr], axis=0)
    records[tag] = data_arr
  return records


def plot_weight_acc_curve(records):
  plot_seq = ["soft_w", "fp_top1", "q_top1"]
  labels = ["$w_q$", r"$Acc_{\mathrm{FP}}$", r"$Acc_{\mathrm{Q}}$"]
  color_seq = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  fig, ax_acc = plt.subplots(figsize=(4, 3), dpi=300)
  ax_w = ax_acc.twinx()
  handles = []
  title = "$w_q$ vs. Training Accuracy"
  for i, tag in enumerate(plot_seq):
    ax = ax_w if "_w" in tag else ax_acc
    alpha = 0.7 if "_w" in tag else 1.0
    data = records[tag]
    h, = ax.plot(data[:, 1], data[:, 2], label=labels[i], color=color_seq[i], alpha=alpha)
    handles.append(h)
  ax_w.legend(handles=handles, loc="lower right")
  ax_acc.set_title(title)
  ax_acc.set_xlabel("training steps")
  ax_acc.set_ylabel("Top-1 acc")
  ax_w.set_ylabel("$w_q$", color=color_seq[0])
  ax_w.tick_params(axis="y", labelcolor=color_seq[0])
  ax_acc.grid(axis="y", linestyle="--")
  fig.tight_layout()
  return fig


if __name__ == "__main__":
  records = load_run_data()
  fig = plot_weight_acc_curve(records)
  fig.savefig(f"wq_acc_curve.pdf")

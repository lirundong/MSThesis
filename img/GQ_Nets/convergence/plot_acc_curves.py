import json
import glob
import re
import os
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_tf_log(path):
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
    data_arr = np.array(data)
  if data_arr[-1, 1] < data_arr[-2, 1]: # abnormal drop at last point
    data_arr = data_arr[:-1, ...]
  return data_arr


def load_runs_logs():
  """
  return (dict): 3-level dict
    train: (phase)
      fp: (exp_type)
        w5a5: (tag)
        w4a4:
        ...
      quant:
        ...
    val: (phase)
      ...
  """
  json_files = glob.glob("./run_acc/*.json")
  records = defaultdict(lambda: defaultdict(OrderedDict))
  for json_file in json_files:
    tag = exp_type = phase = None
    json_name = os.path.split(json_file)[-1].replace(".json", "")
    for token in json_name.split("-"):
      if re.match(r"[wW]\d[aA]\d", token):
        tag = token
      if re.match(r"(quant|fp)_top1_.*", token):
        ts = token.split("_")
        exp_type = ts[0]
        phase = ts[-1]
    assert all([tag, exp_type, phase])
    data_arr = load_tf_log(json_file)
    records[phase][exp_type][tag] = data_arr
  return records


def plot_acc_curves(records):
  figs = OrderedDict()
  for i, phase in enumerate(records.keys()):
    sorted_exp_type = sorted(records[phase].keys())
    for j, exp_type in enumerate(sorted_exp_type):
      sorted_tags = sorted(records[phase][exp_type])
      handles = []
      fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
      title = f"res20 {phase} {exp_type} acc"
      for tag in sorted_tags:
        data_arr = records[phase][exp_type][tag] # [time_stamp, step, value]
        x = data_arr[:, 1]
        y = data_arr[:, 2]
        h, = ax.plot(x, y, label=tag)
        handles.append(h)
      ax.legend(handles=handles)
      ax.set_title(title.title())
      ax.set_xlabel("training steps")
      ax.set_ylabel("Top-1 acc")
      ax.grid(axis="y", linestyle="--")
      fig.tight_layout()
      figs[title.replace(" ", "_")] = fig
  return figs

if __name__ == "__main__":
  records = load_runs_logs()
  figs = plot_acc_curves(records)
  for name, fig in figs.items():
    fig.savefig(f"{name}.pdf")

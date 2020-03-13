import json
import glob
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_run_logs():
  """
  Return (dict):
    "layer_name": (layer)
      "a_lb" (param):
        "w4a4" (tag): <param array of step, value>
        "w3a3": <param array of step, value>
        ...
      "a_ub" (param):
      ...
  """
  records = defaultdict(lambda: defaultdict(OrderedDict))
  for json_file in glob.glob("./run_q_bounds/*.json"):
    json_name = os.path.split(json_file)[-1].replace(".json", "")
    layer = param = tag = None
    for token in json_name.split("-"):
      if re.match(r"[wW]\d[aA]\d", token):
        tag = re.match(r"[wW]\d[aA]\d", token).group(0)
      if re.match(r"[wa]_[lu]b", token):
        match = re.match(r"[wa]_[lu]b", token)
        param = match.group(0)
        layer = token[match.end(0) + 1:]
    assert all([layer, param, tag])
    with open(json_file, "r", encoding="utf-8") as f:
      data = json.load(f)
      data_arr = np.array(data)
      records[layer][param][tag] = data_arr
  return records


def plot_q_bounds(records):
  figs = OrderedDict()
  bounds_seq = [("act", "a_lb", "a_ub"), ("weight", "w_lb", "w_ub")]
  color_seq = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  for layer_name, layer_data in records.items():
    for p_type, lb_name, ub_name in bounds_seq:
      fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
      title = f"{layer_name} {p_type} bounds"
      sorted_tags = sorted(layer_data[lb_name].keys())
      handles = []
      for i, tag in enumerate(sorted_tags):
        color = color_seq[i]
        lb_data = layer_data[lb_name][tag]
        ub_data = layer_data[ub_name][tag]
        lb_handle, = ax.plot(lb_data[:, 1], lb_data[:, 2], color=color, label=f"{tag} lb")
        ub_handle, = ax.plot(ub_data[:, 1], ub_data[:, 2], color=color, linestyle="--", label=f"{tag} ub")
        handles += [lb_handle, ub_handle]
      ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc="upper left")
      ax.set_xlabel("training steps")
      ax.set_ylabel("bound value")
      ax.set_title(title.title())
      ax.grid(axis="y", linestyle="--")
      fig.tight_layout()
      figs[title.replace(" ", "_")] = fig
  return figs


if __name__ == "__main__":
  records = load_run_logs()
  figs = plot_q_bounds(records)
  for fig_name, fig in figs.items():
    fig.savefig(f"{fig_name}.pdf")

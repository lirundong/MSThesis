# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"


def get_act_pairs(archive, layer_name=None):

  def _flat_tensor(tensor):
    assert tensor.ndim >= 2
    return tensor[0].reshape(-1)
  
  act_pairs = OrderedDict()
  for name, tensor in archive.items():
    if layer_name is not None and layer_name not in name:
      continue
    name, suffix = name.split("_")
    if name in act_pairs:
      continue
    another_suffix = "q" if suffix == "fp" else "fp"
    act_pair = {
      suffix: _flat_tensor(tensor),
      another_suffix: _flat_tensor(archive[f"{name}_{another_suffix}"]),
    }
    act_pairs[name] = act_pair
  return act_pairs


def plot_overlapped_hist(act_pairs, fig_path):
  n = len(act_pairs)
  fig, axs = plt.subplots(n, figsize=(4, 3 * n))
  if n < 2:
    axs = (axs, )
  for i, (name, act_pair) in enumerate(act_pairs.items()):
    ax = axs[i]
    ax.title.set_text(f"{name} Act. Distributions")
    fp_tensor = act_pair["fp"]
    q_tensor = act_pair["q"]
    ax.grid(axis="y", linestyle="--")
    ax.hist(fp_tensor, bins=128, label=r"$x_f$")
    ax.hist(q_tensor, bins=128, label=r"$x_q$")
    ax.legend()
    ax.set_xlabel(r"$x_{f, q}$ value")
    ax.set_ylabel("Frequency")
  fig.tight_layout()
  fig.savefig(fig_path)
  print(f"histogram has saved to: {fig_path}")


if __name__ == "__main__":
  parser = ArgumentParser("Plot histograms of QNN activations")
  parser.add_argument("-i", "--input", help="path of input archive")
  parser.add_argument("-o", "--output", help="path of output figure")
  parser.add_argument("-n", "--name", help="name of target layer")
  args = parser.parse_args()

  archive = np.load(args.input)
  act_pairs = get_act_pairs(archive, args.name)
  plot_overlapped_hist(act_pairs, args.output)

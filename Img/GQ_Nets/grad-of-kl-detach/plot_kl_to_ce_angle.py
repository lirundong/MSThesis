import json

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"


def load_file(fname, drop=None):
  with open(fname, "r", encoding="utf-8") as f:
    data = np.array(json.load(f))
    if drop is not None:
      data = data[:drop]
    return data

def plot_xy(ax, title, xlabel, ylabel, fmts, with_legend=True, **kwargs):
  legend_handles = []
  for name, data in kwargs.items():
    timestamps = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    fmt = fmts[name] 
    h, = ax.plot(x, y, fmt, label=name)
    legend_handles.append(h)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(axis="y", linestyle="--")
  if with_legend:
    ax.legend(handles=legend_handles, loc="lower right")

if __name__ == "__main__":
  ce_to_kl_detached = load_file("run-resnet20-cifar10-detach-kl-logs-tag-ce_loss_to_kl_loss_layer3.2.conv2.json",
                                drop=-1)
  ce_to_kl_no_detach = load_file("run-resnet20-cifar10-no-detach-kl-logs-tag-ce_loss_to_kl_loss_layer3.2.conv2.json")
  top1_fp_detached = load_file("run-resnet20-cifar10-detach-kl-logs-tag-fp_top1_eval_acc_val.json", drop=-1)
  top1_q_no_detach = load_file("run-resnet20-cifar10-no-detach-kl-logs-tag-fp_top1_eval_acc_val.json")
  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
  plot_xy(ax1, 
          title=r"$D_{\cos} \left( \nabla_{W_L}\mathcal{L}_f, \nabla_{W_L}\mathcal{L}_q \right)$",
          xlabel="steps", ylabel="$\cos$ distance",
          fmts={"detached": "-", "non_detached": "-."},
          with_legend=False,
          detached=ce_to_kl_detached,
          non_detached=ce_to_kl_no_detach,)
  plot_xy(ax2, 
          title=r"Evaluation accuracy",
          xlabel="steps", ylabel="Top-1 accuracy",
          fmts={"detached": "-", "non_detached": "-."},
          detached=top1_fp_detached,
          non_detached=top1_q_no_detach,)
  plt.tight_layout()
  plt.savefig("kl_to_ce_angle.pdf")

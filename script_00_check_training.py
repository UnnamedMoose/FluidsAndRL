import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os

import resources

save_dir = "./trainingData_0_serialTraining"

matplotlib.rc("font", **{"family": "serif", "weight": "normal", "size": 15})
matplotlib.rcParams["figure.figsize"] = (9, 8)

eval_data, eval_summary = resources.read_monitor(
    os.path.join(save_dir, "eval_monitor.monitor.csv"), n_evals_per_step=3)

fig, ax = resources.plot_training(eval_data, eval_summary)
plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=200, bbox_inches="tight")

plt.show()



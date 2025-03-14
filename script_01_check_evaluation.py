import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os

import resources

matplotlib.rc("font", **{"family": "serif", "weight": "normal", "size": 15})
matplotlib.rcParams["figure.figsize"] = (9, 8)

eval_dir = "evaluationData_0_deterministic_vs_sbl_serial"

dat = np.load(os.path.join(eval_dir, "evaluation_rewards.npz"))
allRewards_det, allRewards_rl = dat["allRewards_det"], dat["allRewards_rl"]
n_eval_episodes = len(allRewards_rl)

histories_rl = []
tot_rew_rl = []
for i in range(n_eval_episodes):
    histories_rl.append(pandas.read_csv(os.path.join(eval_dir, f"eval_hist_rl_{i:d}.csv")))
    tot_rew_rl.append(histories_rl[-1]["reward"].sum())

histories_det = []
tot_rew_det = []
for i in range(n_eval_episodes):
    histories_det.append(pandas.read_csv(os.path.join(eval_dir, f"eval_hist_det_{i:d}.csv")))
    tot_rew_det.append(histories_rl[-1]["reward"].sum())

iep = np.array(range(len(allRewards_det))) + 1

# Compare stats.
fig, ax = resources.niceFig("Episode", "Final reward")
ax.bar(iep-0.4, allRewards_det, 0.4, align="edge", color="b", label="Deterministic")
ax.bar(iep, allRewards_rl, 0.4, align="edge", color="r", label="RL")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
plt.savefig(os.path.join(eval_dir, "evaluation_rewards_final.png"), dpi=200, bbox_inches="tight")

fig, ax = resources.niceFig("Episode", "Total reward")
ax.bar(iep-0.4, tot_rew_det, 0.4, align="edge", color="b", label="Deterministic")
ax.bar(iep, tot_rew_rl, 0.4, align="edge", color="r", label="RL")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
plt.savefig(os.path.join(eval_dir, "evaluation_rewards_total.png"), dpi=200, bbox_inches="tight")

plt.show()




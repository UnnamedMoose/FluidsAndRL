import matplotlib
import matplotlib.pyplot as plt
import subprocess
import socket
import time
import os
import numpy as np
import pandas
import sb3_contrib

import resources

n_eval_episodes = 10
eval_dir = "evaluationData_0_deterministic_vs_sbl_serial"

# Prepare output dir.
os.makedirs(eval_dir, exist_ok=True)

# Evaluate using RL agent.
model_name = "./trainingData_0_serialTraining/modelSaves/best_model.zip"
env_eval = resources.WLEnv("sim_02_swimmerClient.jl", os.path.join(eval_dir, "episode_rlEval_sb3_serial"), n_threads=8)
agent = sb3_contrib.TQC.load(
    model_name,
    env=env_eval,
)
agent.set_parameters(model_name)
print("Evaluating RL")
try:
    mean_reward_rl, allRewards_rl, histories_rl, _ = resources.evaluate_agent(
        agent, env_eval, num_episodes=n_eval_episodes, seed=123, reset_kwargs={})
finally:
    env_eval.close()

# Evaluate using a deterministic swimmer.
agent = resources.DummySwimmerAgent()
env_eval = resources.WLEnv("sim_02_swimmerClient.jl", os.path.join(eval_dir, "episode_rlEval_deterministic"), n_threads=8)
print("Evaluating deterministic")
try:
    mean_reward_det, allRewards_det, histories_det, _ = resources.evaluate_agent(
        agent, env_eval, num_episodes=n_eval_episodes, seed=123, reset_kwargs={})
finally:
    env_eval.close()

# Save data.
np.savez(
    os.path.join(eval_dir, "evaluation_rewards.npz"),
    allRewards_det=allRewards_det,
    allRewards_rl=allRewards_rl
)
for i, hist in enumerate(histories_rl):
    hist.to_csv(os.path.join(eval_dir, f"eval_hist_rl_{i:d}.csv"), index=False)
for i, hist in enumerate(histories_det):
    hist.to_csv(os.path.join(eval_dir, f"eval_hist_det_{i:d}.csv"), index=False)


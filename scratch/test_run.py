import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = "C:\\Users\\lebat\\Documents\\Github\\xbone-net"

cmd = [
    "C:\\Users\\lebat\\miniconda3\\envs\\Thesis\\python.exe",
    "train.py",
    "--config-name=experiment/btxrd/proposed/ours_xbone_net",
    "seed=42",
    "params.phase1.epochs=1",
    "params.phase2.epochs=15",
    "params.phase2.lr=2e-4"
]

subprocess.run(cmd, env=env)

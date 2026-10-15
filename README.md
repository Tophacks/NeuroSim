# NeuroSim

Open-source reinforcement learning experiments in physics-based environments. First experiment: Gymnasium LunarLander-v3 with PPO.

## Windows / Cursor quick start

Install Python 3.11, open this folder in Cursor, then run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --steps 500000 --preview-every 10000
python watch.py --episodes 5
```

If activation is blocked, use `.venv\Scripts\python.exe` instead of `python` for the install and run commands. The preview window appears every 10,000 training steps and pauses training while the preview runs. Close the window normally after the preview ends. To train without previews, pass `--preview-every 0`.

## Files

- `train.py`: PPO training, periodic visual evaluation, and checkpoints.
- `watch.py`: evaluate a saved model with live rendering.
- `requirements.txt`: Python dependencies.

The training environment uses Box2D physics via Gymnasium; NeuroSim does not implement a new physics engine. Training results vary and 500,000 steps do not guarantee reliable landings.

## Next experiments

Wind and disturbance robustness, reward comparison, evaluation across fixed seeds, and MuJoCo robotics environments.

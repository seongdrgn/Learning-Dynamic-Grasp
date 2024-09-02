# Learning Dynamic Grasp

## Simulation Environment (Isaac-Lab)

1. Pretraining default policy via Asymmetric Actor-Critic

   ```python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-rl_games --num_envs=2048 --headless```

3. Training "Goal Estimator" for predicting goal position of the thrown object

   ```python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-Goal-Estimator --num_envs=2048 --headless --enable_cameras```
   
   "Goal Estimator"'s inputs are initial depth images and outputs goal position of the thrwon object

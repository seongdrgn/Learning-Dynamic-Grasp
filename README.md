# Learning Dynamic Grasp

## Simulation Environment (Isaac-Lab)

Pretraining default policy via asymmetric actor-critic

* Training
  
   ```python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-rl_games --num_envs=2048 --headless```

* Evaluation
  
   ```python source/standalone/workflows/rl_games/play.py --taks Isaac-Dynamic-Catch-rl_games --num_envs=1 --checkpoint=/your_root/logs/rl_games/dynamic_catch_asym/2024-08-22_10-58-12/nn/dynamic_catch_asym.pth```

Training "Goal Estimator" for predicting goal position of the thrown object using pretrained policy

* Training

   ```python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-Goal-Estimator --num_envs=2048 --headless --enable_cameras```
   
   "Goal Estimator"'s inputs are initial depth images and outputs goal position of the thrwon object

## Troubleshooting

For anaconda users

> Users must complete installation the Isaac-Lab
```
cd /your/workspace/IsaacLab/_isaac_sim && source setup_conda_env.sh
```

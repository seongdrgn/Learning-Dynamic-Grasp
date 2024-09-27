# Learning Dynamic Grasp

## Simulation Environment (Isaac-Lab)
<img src="https://github.com/user-attachments/assets/816f6280-6165-4b7d-8148-caf67805492d" width="600" height="312"> 

### Pretraining default policy via asymmetric actor-critic
> Environment version : ```dynamic_catch_v1.py```

* Training
  
   ```ruby
  python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-rl_games --num_envs=2048 --headless
   ```

* Evaluation
  
   ```ruby
  python source/standalone/workflows/rl_games/play.py --taks Isaac-Dynamic-Catch-rl_games --num_envs=1 --checkpoint=/your_root/logs/rl_games/dynamic_catch_asym/2024-08-22_10-58-12/nn/dynamic_catch_asym.pth
   ```

### Training "Goal Estimator" 
Goal estimator predicts goal position of the thrown object using pretrained policy.
> Environment version : ```dynamic_catch_v2.py```

* Training

   ```ruby
  python source/standalone/workflows/rl_games/train.py --task Isaac-Dynamic-Catch-Goal-Estimator --num_envs=2048 --headless --enable_cameras
   ```
   
   "Goal Estimator"'s inputs are sequential depth images for t timesteps and outputs goal position of the thrwon object

## Real Environment

On going project...

## Troubleshooting

For anaconda users

> Users must complete installation the Isaac-Lab
```
cd /your/workspace/IsaacLab/_isaac_sim && source setup_conda_env.sh
```

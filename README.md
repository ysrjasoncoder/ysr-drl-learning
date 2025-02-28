# ysr-drl-learning

ddpg.py is a parameter adjustment of ddpg.py in the simple-pytorch-rl repository, trying to get a better reward.

ddpg-my.py is a rewrite of ddpg, using Soft updates to target networks. That's the only change.

mbrl.py is a model-based reinforcement learning based on ddpg.py in the simple-pytorch-rl repository. It basically refers to the papers I read before. It does not improve the final reward, but it can accelerate convergence and get results in an earlier episode.

sac.py is a sac code that I tried to read and run. It can get a better reward.

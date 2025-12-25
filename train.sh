#!/bin/bash
export PYTHONPATH=.

# /home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 1 --policy_type mlp --value_type mlp

# /home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 2 --policy_type mlp --value_type mlp

/home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 1 --policy_type dmoe --value_type dmoe

/home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 2 --policy_type dmoe --value_type dmoe

# /home/hzq/anaconda3/envs/moevs/bin/python src/ppo.py --envs_model change --seed 1 --policy_type smoe --value_type smoe

# /home/hzq/anaconda3/envs/moevs/bin/python src/ppo.py --envs_model change --seed 2 --policy_type smoe --value_type smoe

# /home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 1 --policy_type moe --value_type moe

# /home/labliu/anaconda3/envs/plasticine/bin/python src/ppo.py --envs_model change --seed 2 --policy_type moe --value_type moe

wait
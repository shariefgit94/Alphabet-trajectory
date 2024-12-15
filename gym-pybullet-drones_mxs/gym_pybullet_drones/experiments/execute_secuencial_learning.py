#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage1
from gym_pybullet_drones.experiments.learning_script import run_learning

print("############# WCL-DISTURBANCES-RANDOM-VELOCITIES_ACC #############")

results = run_learning(environment=ObS12Stage1,
                       learning_id="WCL-RANDOM-VELOCITIES_ACC",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(30e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=True, threshold=600.),
                       save_checkpoints=dict(save=True, save_frequency=25000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

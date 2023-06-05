import numpy as np
import time
from RainbowDQNAgent import RainbowDQNAgent
from utils import plot_learning_curve, make_env

env = make_env("BreakoutNoFrameskip-v4")
env_name = "BreakoutNoFrameskip-v4"
max_n_steps = 500000
algo = "RainbowDQN"
eps_better = 50
avg_human = 35411
avg_random = 10781
avg_best_linear = 23411


# env = make_env("PongNoFrameskip-v4")
# env_name = "PongNoFrameskip-v4"
# max_n_steps = 500000
# algo = "RainbowDQN"
# eps_better = 0.5
# avg_human = 9.3
# avg_random = -20.7
# avg_best_linear = -19

# env = make_env("ALE/CrazyClimber-v5")
# env_name = "CrazyClimber-v5"
# best_score = -np.inf
# n_games = 10000
# max_n_steps = 1000000
# algo = "DuelingDQN"
# eps_min = 0.1
# eps_better = 50
# avg_human = 35411
# avg_random = 10781
# avg_best_linear = 23411

start = time.time()

agent = RainbowDQNAgent(env=env, buffer_size=50000, batch_size=32, algo=algo, env_name=env_name, chkpt_dir="models/")

plot_name = algo + '_' + env_name + '_' + str(max_n_steps) + 'steps'
figure_file = 'plots/' + plot_name + '.png'

scores, steps, losses = agent.train(max_n_steps, eps_better)


plot_learning_curve(steps, scores, figure_file, avg_human, avg_random, avg_best_linear)

end = time.time()

train_time = end - start

print('Train time:', train_time, 'seconds')

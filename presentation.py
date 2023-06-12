from RainbowDQNAgent import RainbowDQNAgent
from utils import make_env
import gymnasium as gym

env = make_env("BreakoutNoFrameskip-v4")
env_name = "BreakoutNoFrameskip-v4"
algo = "RainbowDQN"
add = "1kk"

env = gym.wrappers.RecordVideo(env, video_folder='./videos', name_prefix="BreakoutNoFrameskip-v4")

agent = RainbowDQNAgent(env=env, buffer_size=50000, batch_size=128, algo=algo, env_name=env_name, chkpt_dir="models/", pres=True, model_name_add=add)

agent.load_models()
# agent.load_models_named('most_recent')

observation, info = env.reset()
max_lives = info['lives']
terminate_on_life_lose = False
done = False
autofire = True
cnt = 0
while not done:
    if autofire and cnt % 100 == 0:
        observation, reward, terminated, truncated, info = env.step(1)
    cnt += 1
    action = agent.select_action(observation)
    prev_action = action
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if terminate_on_life_lose and info['lives'] != max_lives:
        done = True

env.close()


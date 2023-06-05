from RainbowDQNAgent import RainbowDQNAgent
from utils import make_env

env = make_env("CrazyClimberNoFrameskip-v4", True)
env_name = "CrazyClimberNoFrameskip-v4"
algo = "RainbowDQN"

agent = RainbowDQNAgent(env=env, buffer_size=50000, batch_size=128, algo=algo, env_name=env_name, chkpt_dir="models/", pres=True)

agent.load_models()

observation, info = env.reset()
done = False

while not done:
    action = agent.select_action(observation)
    prev_action = action
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

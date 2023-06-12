import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.core import ObsType, WrapperObsType
from segment_tree import MinSegmentTree, SumSegmentTree
from collections import deque
import random


def plot_learning_curve(x, scores, filename, avg_human, avg_random, avg_best_linear):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    scores_len = len(scores)
    running_avg = np.empty(scores_len)
    for i in range(scores_len):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    ax.plot(x, running_avg, color="C1")
    ax.yaxis.tick_right()
    ax.set_ylabel('Score', color="C1")
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', colors="C1")

    plt.axhline(y=avg_human, color='C2', linestyle='-', label="avg_human")
    plt.axhline(y=avg_random, color='C3', linestyle='-', label="avg_random")
    plt.axhline(y=avg_best_linear, color='C4', linestyle='-', label="avg_best_linear")
    plt.legend(bbox_to_anchor=(1.0075, 1.15), ncol=3)

    plt.savefig(filename)


def plot_loss(losses, filename):
    plt.plot(losses)
    plt.savefig(filename)


def make_env(env_name, human=False, frameskip=1):
    # we create game environment
    if human:
        env = gym.make(env_name, render_mode="human", full_action_space=False, frameskip=frameskip)
    else:
        env = gym.make(env_name, render_mode="rgb_array", full_action_space=False, frameskip=frameskip)
    # chose "better" frame
    env = TakeMaxFrame(env)
    # resize our observation
    env = gym.wrappers.ResizeObservation(env, 84)
    # convert our observation to grayscale
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    #custom wrapper to adjust observastion
    env = ReshapeObservationSpaceWrapper(env, shape=(84, 84))
    # we stack last 4 frames
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


class ReshapeObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ReshapeObservationSpaceWrapper, self).__init__(env)
        self.shape = shape
        #reshape observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation: ObsType) -> WrapperObsType:
        #reshape observation
        observation = observation.reshape(*self.shape)
        #scale values
        observation = observation / 255.0
        return observation


class TakeMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super(TakeMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        combined_reward = 0.0
        # we repeat each action for 4 steps
        terminated = False
        truncated = False
        for i in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            combined_reward += reward
            id = i % 2
            # we keep last 2 frames
            self.frame_buffer[id] = observation
            if terminated or truncated:
                break
        # we return max frame
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, combined_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = observation
        return observation


class ReplayBuffer(object):
    def __init__(self, input_shape, buffer_size, batch_size=32, n_step=1, gamma=0.99):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_ptr = 0
        self.size = 0
        # we keep data in numpy arrays
        self.state_buffer = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.buffer_size, dtype=np.bool)
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    # adding transition to buffer
    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.state_buffer[self.buffer_ptr] = state
        self.action_buffer[self.buffer_ptr] = action
        self.reward_buffer[self.buffer_ptr] = reward
        self.next_state_buffer[self.buffer_ptr] = next_state
        self.terminal_buffer[self.buffer_ptr] = done
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

        return self.n_step_buffer[0]

    # sampling from buffer
    def sample_random_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            states=self.state_buffer[idxs],
            next_states=self.next_state_buffer[idxs],
            actions=self.action_buffer[idxs],
            rewards=self.reward_buffer[idxs],
            dones=self.terminal_buffer[idxs],
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs):
        # for N-step Learning
        return dict(
            states=self.state_buffer[idxs],
            next_states=self.next_state_buffer[idxs],
            actions=self.action_buffer[idxs],
            rewards=self.reward_buffer[idxs],
            dones=self.terminal_buffer[idxs],
        )

    def _get_n_step_info(self, n_step_buffer, gamma):
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, input_shape: int, buffer_size: int, batch_size: int = 32,
                 alpha: float = 0.5, n_step: int = 1, gamma: float = 0.99):
        super(PrioritizedReplayBuffer, self).__init__(
            input_shape, buffer_size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state, action, reward, next_state, done):
        transition = super().store(state, action, reward, next_state, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

        return transition

    def sample_batch(self, beta: float = 0.4):
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        states = self.state_buffer[indices]
        next_states = self.next_state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.terminal_buffer[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx: int, beta: float):
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

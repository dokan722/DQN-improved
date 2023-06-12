import numpy as np
import torch
from RainbowDQNNetwork import RainbowDQNNetwork
from utils import PrioritizedReplayBuffer, ReplayBuffer
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class RainbowDQNAgent(object):
    def __init__(self, env, buffer_size, batch_size, replace=1000, gamma=0.99, lr=0.0001,
                 alpha=0.5, beta=0.4, prior_eps=1e-6,
                 v_min=-10.0, v_max=10.0, atom_size=51, n_step=3,
                 algo=None, env_name=None, chkpt_dir='tmp/dqn', pres=False, model_name_add=""):
        self.chkpt_dir = chkpt_dir

        input_dims = env.observation_space.shape
        n_actions = env.action_space.n
        self.n_actions = n_actions

        self.env = env
        self.batch_size = batch_size
        self.replace = replace
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(input_dims, buffer_size, batch_size, alpha=alpha)

        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(input_dims, buffer_size, batch_size, n_step=n_step, gamma=gamma)

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        eval_name = algo + "_" + env_name + "_dqn" + model_name_add
        self.dqn = RainbowDQNNetwork(input_dims, n_actions, self.atom_size, self.support, eval_name, self.chkpt_dir).to(self.device)

        next_name = algo + "_" + env_name + "_dqn_target" + model_name_add
        self.dqn_target = RainbowDQNNetwork(input_dims, n_actions, self.atom_size, self.support, next_name, self.chkpt_dir).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.transition = list()

        self.max_lives = 0
        self.life_prev = 0
        self.terminate_on_life_lose = False
        self.force_fire = False
        self.no_fire_ctr = 0
        self.is_test = pres

        self.first_gain = True
        self.stolen = 0

    def select_action(self, state):
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(torch.FloatTensor(np.array(state)).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        if self.force_fire and selected_action != 1:
            self.no_fire_ctr += 1
            if self.no_fire_ctr > 100:
                selected_action = 1
                self.no_fire_ctr = 0

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def select_random_action(self, state):
        action = self.env.action_space.sample()
        if not self.is_test:
            self.transition = [state, action]
        return action

    def step(self, action: np.ndarray):
        next_state, reward, done, _, info = self.env.step(action)
        if self.terminate_on_life_lose:
            if info['lives'] != self.max_lives:
                done = True
            if self.first_gain:
                self.stolen += reward
                reward = 0
                self.first_gain = False

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_steps, eps_better, terminate_on_life_lose=False, force_fire=False):
        self.is_test = False
        self.terminate_on_life_lose = terminate_on_life_lose
        self.force_fire = force_fire

        state, info = self.env.reset()
        self.max_lives = info['lives']
        self.life_prev = info['lives']

        update_cnt = 0
        losses = []
        scores = []
        steps_list = []
        score = 0
        episode_counter = 1
        best_score = -np.inf

        for frame_idx in range(1, num_steps + 1):
            if frame_idx < 20000:
                action = self.select_random_action(state)
            else:
                action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            fraction = min(frame_idx / num_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if done:
                avg_score = np.mean(scores[-100:])
                print('episode: ', episode_counter, 'score: ', score,
                      ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                      'episode loss %.5f' % (sum(losses) / max(len(losses), 1)), 'steps', frame_idx)
                if avg_score > best_score + eps_better and frame_idx > 50000:
                    self.save_models()
                    best_score = avg_score
                state, _ = self.env.reset()
                scores.append(score)
                steps_list.append(frame_idx)
                score = 0
                episode_counter += 1
                losses = []

                score += self.stolen
                self.stolen = 0
                self.first_gain = True

            # if training is ready
            if len(self.memory) >= self.batch_size and frame_idx > 20000:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.replace == 0:
                    self._target_hard_update()
            if frame_idx % 10000 == 0:
                self.save_models_named('most_recent')
        self.env.close()
        return scores, steps_list, losses

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _compute_dqn_loss(self, samples, gamma):
        device = self.device
        states = torch.FloatTensor(samples["states"]).to(device)
        next_states = torch.FloatTensor(samples["next_states"]).to(device)
        actions = torch.LongTensor(samples["actions"]).to(device)
        rewards = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(samples["dones"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_states).argmax(1)
            next_dist = self.dqn_target.dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + (1 - dones) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.dqn.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    #saving models
    def save_models(self):
        self.dqn.save_checkpoint()
        self.dqn_target.save_checkpoint()

    def save_models_named(self, name):
        self.dqn.save_checkpoint_named(name)
        self.dqn_target.save_checkpoint_named(name)

    #load models
    def load_models(self):
        self.dqn.load_checkpoint()
        self.dqn_target.load_checkpoint()

    def load_models_named(self, name):
        self.dqn.load_checkpoint_named(name)
        self.dqn_target.load_checkpoint_named(name)


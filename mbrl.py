import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with Model-Based RL')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
parser.add_argument(
    '--planning-horizon',
    type=int,
    default=5,
    metavar='H',
    help='planning horizon for model-based rollouts (default: 5)')
parser.add_argument(
    '--model-rollouts',
    type=int,
    default=5,
    metavar='R',
    help='number of model rollouts per real step (default: 5)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward', 'model_loss'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])


class DynamicsModel(nn.Module):
    """Neural network to learn the dynamics of the environment"""
    
    def __init__(self):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(4, 200)  # state + action
        self.fc2 = nn.Linear(200, 200)
        self.state_head = nn.Linear(200, 3)  # predict next state
        self.reward_head = nn.Linear(200, 1)  # predict reward
        
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        return next_state, reward


class ActorNet(nn.Module):
    """Policy network"""

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = 2.0 * F.tanh(self.mu_head(x))
        return u


class CriticNet(nn.Module):
    """Value network"""

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(4, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1)))
        state_value = self.v_head(x)
        return state_value


class Memory():
    """Replay buffer for storing transitions"""

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class ModelBasedAgent():
    """Model-based RL agent"""

    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 1.0
        # Value function networks
        self.eval_cnet, self.target_cnet = CriticNet().float(), CriticNet().float()
        # Policy networks
        self.eval_anet, self.target_anet = ActorNet().float(), ActorNet().float()
        # Dynamics model
        self.dynamics_model = DynamicsModel().float()
        
        # Replay buffer
        self.memory = Memory(2000)
        
        # Optimizers
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)
        self.optimizer_model = optim.Adam(self.dynamics_model.parameters(), lr=1e-3)

    def select_action(self, state):
        """Select action using the policy network with exploration noise"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float))
        action = dist.sample()
        action.clamp(-2.0, 2.0)
        return (action.item(),)

    def save_param(self):
        """Save model parameters"""
        torch.save(self.eval_anet.state_dict(), 'param/mb_anet_params.pkl')
        torch.save(self.eval_cnet.state_dict(), 'param/mb_cnet_params.pkl')
        torch.save(self.dynamics_model.state_dict(), 'param/mb_dmodel_params.pkl')

    def store_transition(self, transition):
        """Store transition in the replay buffer"""
        self.memory.update(transition)

    def update_dynamics_model(self, batch_size=64):
        """Update the dynamics model using supervised learning"""
        if not self.memory.isfull:
            return 0.0
            
        transitions = self.memory.sample(batch_size)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)
        
        # Update dynamics model
        self.optimizer_model.zero_grad()
        pred_s_, pred_r = self.dynamics_model(s, a)
        
        # Compute loss for state prediction and reward prediction
        state_loss = F.mse_loss(pred_s_, s_)
        reward_loss = F.mse_loss(pred_r, r)
        model_loss = state_loss + reward_loss
        
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.max_grad_norm)
        self.optimizer_model.step()
        
        return model_loss.item()

    def generate_imagined_trajectories(self, s, batch_size=32):
        """Generate imagined trajectories using the dynamics model"""
        states = s.repeat(1, 1) if s.shape[0] == batch_size else s.repeat(batch_size // s.shape[0] + 1, 1)[:batch_size]
        total_rewards = torch.zeros(batch_size, 1)
        
        imagined_states = [states]
        imagined_actions = []
        imagined_rewards = []
        
        for h in range(args.planning_horizon):
            # Select actions using the policy
            actions = self.eval_anet(states)
            
            # Add exploration noise
            noise = torch.normal(mean=0, std=self.var, size=actions.shape)
            noisy_actions = (actions + noise).clamp(-2.0, 2.0)
            
            # Predict next states and rewards using the dynamics model
            next_states, step_rewards = self.dynamics_model(states, noisy_actions)
            
            # Store for later use
            imagined_states.append(next_states)
            imagined_actions.append(noisy_actions)
            imagined_rewards.append(step_rewards)
            
            # Update for next step
            states = next_states
            total_rewards += step_rewards * (args.gamma ** h)
        
        return imagined_states, imagined_actions, imagined_rewards, total_rewards

    def update_policy_with_model_rollouts(self, start_states):
        """Update policy using model rollouts"""
        # Generate imagined trajectories using the dynamics model
        imagined_states, imagined_actions, imagined_rewards, total_rewards = self.generate_imagined_trajectories(start_states)
        
        # Policy gradient update
        self.optimizer_a.zero_grad()
        
        # Use the model to evaluate the expected return of our policy
        a_loss = 0
        discount = 1.0
        
        # Sum up the expected Q-values along the trajectory
        for h in range(args.planning_horizon):
            states = imagined_states[h]
            actions = self.eval_anet(states)  # Get on-policy actions
            q_values = self.eval_cnet(states, actions)
            a_loss -= discount * q_values.mean()
            discount *= args.gamma
        
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

    def update(self):
        """Update policy and value networks"""
        self.training_step += 1

        # Update dynamics model first
        model_loss = self.update_dynamics_model()
        
        if not self.memory.isfull:
            return 0, model_loss
            
        # Sample real transitions for DDPG update
        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        # Regular DDPG update with real transitions
        with torch.no_grad():
            q_target = r + args.gamma * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        # Update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # Regular actor update
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        # Model-based policy improvement
        # Only do model rollouts if the model has been trained enough
        if model_loss < 0.1 and self.training_step > 1000:
            for _ in range(args.model_rollouts):
                # Use a subset of states for diversity
                idx = torch.randperm(len(s))[:8]
                start_states = s[idx]
                self.update_policy_with_model_rollouts(start_states)

        # Update target networks
        if self.training_step % 200 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 201 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())

        # Decay exploration noise
        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item(), model_loss


def main():
    env = gym.make('Pendulum-v1')
    agent = ModelBasedAgent()

    training_records = []
    running_reward, running_q, running_model_loss = -1000, 0, 0
    
    for i_ep in range(1000):
        score = 0
        state, _ = env.reset(seed=args.seed+i_ep)

        for t in range(200):
            action = agent.select_action(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            
            if args.render:
                env.render()
                
            agent.store_transition(Transition(state, action, (reward + 8) / 8, state_))
            state = state_
            
            if agent.memory.isfull:
                q, model_loss = agent.update()
                running_q = 0.99 * running_q + 0.01 * q
                running_model_loss = 0.99 * running_model_loss + 0.01 * model_loss

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward, running_model_loss))

        if i_ep % args.log_interval == 0:
            print('Step {}\tAverage score: {:.2f}\tAverage Q: {:.2f}\tModel Loss: {:.4f}'.format(
                i_ep, running_reward, running_q, running_model_loss))
                
        if running_reward > -150:
            print("Solved! Running reward is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            with open('log/mbrl_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break

    env.close()

    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('Model-Based RL')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    
    plt.subplot(2, 1, 2)
    plt.plot([r.ep for r in training_records], [r.model_loss for r in training_records])
    plt.xlabel('Episode')
    plt.ylabel('Model Loss')
    
    plt.tight_layout()
    plt.savefig("img/mbrl.png")
    plt.show()


if __name__ == '__main__':
    main()
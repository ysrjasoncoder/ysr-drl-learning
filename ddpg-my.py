import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt

# Simple networks
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(3, 64)
        self.layer2 = nn.Linear(64, 1)
        
    def forward(self, state):
        x = F.relu(self.layer1(state))
        action = 2.0 * torch.tanh(self.layer2(x))  #  [-2, 2]
        return action

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(4, 64)  
        self.layer2 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        x = F.relu(self.layer1(torch.cat([state, action], dim=1)))
        value = self.layer2(x)
        return value

class ReplayBuffer:
    def __init__(self, size=2000):
        self.buffer = []
        self.max_size = size
        self.position = 0
        
    def add(self, state, action, reward, next_state):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size=32):
        batch = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states = [], [], [], []
        
        for i in batch:
            state, action, reward, next_state = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards).reshape(-1, 1)),
            torch.FloatTensor(np.array(next_states))
        )
    
    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.noise = 1.0
        
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.buffer = ReplayBuffer()
        self.training_steps = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        action = action + np.random.normal(0, self.noise, size=action.shape)
        action = np.clip(action, -2.0, 2.0)
        
        return action
        
    def update(self):
        if self.buffer.size() < 32:
            return 0
            
        self.training_steps += 1
        
        states, actions, rewards, next_states = self.buffer.sample() #Replay
        
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = rewards + self.gamma * self.critic_target(next_states, target_actions)
            
        current_q = self.critic(states, actions)
        
        critic_loss = F.smooth_l1_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.training_steps % 10 == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.95 * target_param.data + 0.05 * param.data)
                
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(0.95 * target_param.data + 0.05 * param.data)
                
        self.noise = max(self.noise * 0.999, 0.01)
        
        return current_q.mean().item()

# Training loop
def train():
    env = gym.make('Pendulum-v1')
    agent = DDPGAgent()
    
    rewards_history = []
    avg_reward = -1000
    avg_q = 0
    
    for episode in range(1000):
        state, _ = env.reset(seed=episode)
        episode_reward = 0
        
        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.buffer.add(state, action, (reward + 8) / 8, next_state)  
            
            if agent.buffer.size() >= 100:
                q = agent.update()
                avg_q = 0.99 * avg_q + 0.01 * q
                
            state = next_state
            episode_reward += reward
            
        avg_reward = 0.9 * avg_reward + 0.1 * episode_reward
        rewards_history.append((episode, avg_reward))
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Q: {avg_q:.2f}")
            
        if avg_reward > -200:
            print(f"Solved at episode {episode} with avg reward {avg_reward:.2f}!")
            break
    
    # Plot results
    episodes, rewards = zip(*rewards_history)
    plt.plot(episodes, rewards)
    plt.title("DDPG Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()
    plt.savefig("img/ddpg-my.png")
    env.close()

if __name__ == "__main__":
    train()
import argparse
import gym
import numpy as np
from itertools import count
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()

torch.manual_seed(args.seed)
if not torch.cuda.is_available():
    args.cuda = False

if torch.cuda.is_available() and args.cuda:
    torch.cuda.manual_seed(args.seed)

env = gym.make('Breakout-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 40, kernel_size=3)
        self.fc1 = nn.Linear(4200, 50)
        self.fc2 = nn.Linear(50, 6)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4200)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        action_scores = self.fc2(x)
        return F.softmax(action_scores)


model = Policy()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def select_action(state):
    state = cv2.resize(state, (40, 103))
    state = np.transpose(state, (2, 0, 1))
    print(state.shape)
    state = torch.from_numpy(state).float().unsqueeze(0)
    # Norm
    state /= 255
    if args.cuda:
        state = state.cuda()
    probs = model(Variable(state))
    action = probs.multinomial()
    model.saved_actions.append(action)
    return action.data


def finish_episode():
    R = 0
    rewards = []
    # Weight sum of rewards
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)

    # Norm
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # What'is the action?
    for action, r in zip(model.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


running_reward = 0
for i_episode in count(1):
    state = env.reset()
    total_R = 0.0
    for t in range(10000): # Don't infinite loop while learning
        action = select_action(state)
        new_state, reward, done, _ = env.step(action[0,0])
        # print(new_state-state, reward)
        # print(new_state)
        # print((new_state-state)[103])
        state = new_state
        # print(state, reward)
        # input()
        running_reward = reward
        total_R += reward
        if args.render:
            env.render(mode="human")
        model.rewards.append(total_R)
        if done:
            break
    # running_reward = #running_reward * 0.99 + t * 0.01
    print("fini in")
    finish_episode()
    print("fini out")
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5f}\tAverage length: {:.2f}'.format(
            i_episode, total_R, running_reward))
    if running_reward > 220:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break
    exit()

import numpy as np
import sys
import torch
from torch.nn import Softmax, LogSoftmax, Sigmoid
from torch.autograd import Variable
import random
from four_rooms import FourRoomsEnvironment
import matplotlib.pyplot as plt
from tqdm import tqdm


class Option():
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Policy parameters
        self.theta = Variable(torch.Tensor(
            np.random.rand(n_states, n_actions)), requires_grad=True)
        # Termination parameters
        self.upsilon = Variable(torch.Tensor(
            np.random.rand(n_states)), requires_grad=True)
    
    
    def pi(self, state_index, T=0.1):
        # Input : index in [0, n_states - 1]
        # Return : pi(.|state), variable of shape (1, n_actions)
        state_var = self.varFromStateIndex(state_index)
        probs = Softmax()(torch.matmul(state_var, self.theta) / T)
        return probs
    

    def beta(self, state_index):
        # Input : index in [0, n_states - 1]
        # Return : beta(state), variable of shape (1)
        state_var = self.varFromStateIndex(state_index)
        return Sigmoid()(torch.matmul(state_var, self.upsilon))
    

    def pickAction(self, state_index):
        # Input : index in [0, n_states - 1]
        # Return : one of "left", "up", "right", or "down",
        # index of action chosen in [0, n_actions]
        # and one-hot variable of shape (1, n_actions)
        probs = self.pi(state_index).data.numpy().reshape(-1)
        action_index = np.random.choice(self.n_actions, size=1, p=probs)[0]
        action, action_one_hot = self.actionFromActionIndex(action_index)
        return action, action_index, action_one_hot
    

    def varFromStateIndex(self, state_index):
        # Input : index in [0, n_states - 1]
        # Return : one-hot variable of shape (1, n_states)
        s = np.zeros(self.n_states)
        s[state_index] = 1
        return Variable(torch.Tensor(s)).view(1, -1)
    

    def actionFromActionIndex(self, action_index):
        # Input : index in [0, n_actions - 1]
        # Return : one of "left", "up", "right", or "down"
        # and one-hot variable of shape (1, n_actions)
        if action_index == 0:
            action = "left"
        elif action_index == 1:
            action = "up"
        elif action_index == 2:
            action = "right"
        elif action_index == 3:
            action = "down"
        a = np.zeros(self.n_actions)
        a[action_index] = 1
        return action, Variable(torch.Tensor(a)).view(1, -1)



""" Option-Critic architecture """
class OptionCritic():
    def __init__(self, gamma=0.99, alpha_critic=0.5, alpha_theta=0.25, 
                 alpha_upsilon=0.25, n_options=4):    
        self.gamma = gamma                 # Discount factor
        self.alpha_critic_lr = alpha_critic   # Critic learning rate
        self.alpha_theta_lr = alpha_theta     # Intra-option policies learning rate
        self.alpha_upsilon_lr = alpha_upsilon # Termination functions learning rate
        
        n_states = 13*13
        n_actions = 4
        self.options = [Option(n_states, n_actions) for _ in range(n_options)]
        
        self.current_option = None
        # Keep track of one hot var and index of last action taken
        self.last_action_one_hot = None
        self.last_action_index = None
        
        # Action values in the context of (state, option) pairs
        self.Q_U = np.zeros((n_states, n_options, n_actions)) # Q_U(s,o,a) as in paper
        # Option values (computed from Q_U)
        self.Q_omega = np.zeros((n_states, n_options)) # Q_omega(s,o) as in paper 
        # State values (computed from Q_omega)
        self.V = np.zeros(n_states)
        
    def epsilonGreedyPolicy(self, state_tuple, epsilon=0.01):
        state_index = self.sIdx(state_tuple)
        # If current option is None, pick a new one epsilon greedily; it's set to None when it is termninated 
        if self.current_option is None:
            # epsilon-greedy
            # Pick greedy option with probability (1 - epsilon)
            if random.uniform(0, 1) > epsilon:
                best_option_idx = np.argmax(self.Q_omega[state_index])
                self.current_option = self.options[best_option_idx]
            # Pick random action with probability epsilon
            else:
                self.current_option = random.choice(self.options)

        # Pick action according to current option
        action, action_index, action_one_hot = self.current_option.pickAction(state_index)
        # Record one hot var and index of last action taken
        self.last_action_one_hot = action_one_hot
        self.last_action_index = action_index
        return action
    
    def storeTransition(self, state, reward, next_state):
        pi = self.current_option.pi(self.sIdx(state)) #log pi(.|state), variable of shape (1, n_actions)
        beta = self.current_option.beta(self.sIdx(next_state)) #variable of shape (1); note this is for next step
        
        # 1) Critic improvement
        # Update estimate of Q_U[state, current_option, action]
        self.Option_evaluation(state, reward, next_state, pi, beta)
        
        # 2) Actor improvement
        # Take a gradient step for policy and termination parameters
        # of current option
        self.Option_improvement(state, next_state, pi, beta)
        
        # If current option ends, set current option to None
        beta = self.current_option.beta(self.sIdx(next_state)).data[0]
        if random.uniform(0, 1) < beta:
            self.current_option = None
        
    def Option_evaluation(self, state, reward, next_state, pi, beta):
        # Algorithm 1 paper
        s1 = self.sIdx(state)
        s2 = self.sIdx(next_state)
        o = self.oIdx(self.current_option)
        a = self.last_action_index
        
        # Update Q_U
        beta = beta.data[0]
        target = reward + self.gamma * (1 - beta) * self.Q_omega[s2, o] \
            + self.gamma * beta * np.max(self.Q_omega[s2])  
        self.Q_U[s1, o, a] += \
            self.alpha_critic_lr * (target - self.Q_U[s1, o, a])
            
        # Update Q_omega since Q_U has changed
        # Q_omega(s,o) =  pi_o,theta(a | s) QU(s,o,a); equation 1 of the paper
        self.Q_omega[s1, o] = pi.data.numpy().reshape(-1).dot(self.Q_U[s1, o])
        
        # Update V since Q has changed
        # This update is only valid if the policy over options is greedy
        self.V[s1] = np.max(self.Q_omega[s1, o])
        
    def Option_improvement(self, state, next_state, pi, beta):
        s1 = self.sIdx(state)
        s2 = self.sIdx(next_state)
        o = self.oIdx(self.current_option)
        a = self.last_action_index
        
        # 1) Policy update
        # Compute log pi(last_action_taken | state)
        logprobs = torch.log(pi)
        logprob = torch.sum(logprobs * self.last_action_one_hot)
        # Compute gradient of theta w.r.t this quantity
        logprob.backward()
        grad_theta = self.current_option.theta.grad.data
        # Take a gradient step; see algo 1 in paper
        self.current_option.theta.data += self.alpha_theta_lr * \
            self.Q_U[s1, o, a] * grad_theta
        # Zero gradient
        self.current_option.theta.grad.data.zero_()
        
        # 2) Termination function update
        # Compute gradient of upsilon, w.r.t beta(next_state)
        beta.backward()
        grad_upsilon = self.current_option.upsilon.grad.data
        # Take a gradient step;
        self.current_option.upsilon.data -= self.alpha_upsilon_lr * \
            (self.Q_omega[s2, o] - self.V[s2]) * grad_upsilon
        # Zero gradient
        self.current_option.upsilon.grad.data.zero_()
        
    def sIdx(self, state):
        return state[0] * 13 + state[1]
    
    def oIdx(self, option):
        return self.options.index(option)


env = FourRoomsEnvironment(start_loc=("random"))


def run_episode(verbose=False):
    n_steps = 0
    state = env.reset()
    while True:
        n_steps += 1
        action = agent.epsilonGreedyPolicy(state)
        if verbose:
            print("State = {}, Option = {}, Action = {}".format(
                state, agent.current_option, action))
        next_state, reward, done = env.step(action)
        agent.storeTransition(state, reward, next_state)
        state = next_state
        if done:
            return n_steps
        # If episode takes more than 1000 steps to finish reset it
        if n_steps > 1000:
            n_steps = 0
            state = env.reset()


n_repetitions = 5
n_episodes = 301

average_len_episodes = []

for i in range(n_repetitions):
    print ('%d repetition'%i)
    agent = OptionCritic()
    len_episodes = []
   
    for j in tqdm(range(n_episodes)):
        n_steps = run_episode()
        len_episodes.append(n_steps)
        
    average_len_episodes.append(len_episodes)
    
average_len_episodes = np.array(average_len_episodes).mean(axis=0)
    
plt.plot(range(n_episodes), average_len_episodes)
plt.xlabel("Episodes")
plt.ylabel("Steps per episode")
plt.savefig("plots/option-critic_4options.png")
plt.show()


def visualizeTermination(option):
    beta = []
    for i in range(13):
        row = []
        for j in range(13):
            s = agent.sIdx((i, j))
            row.append(option.beta(s).data[0])
        beta.append(row)
    beta = np.array(beta) * (env.grid >= 0)
    plt.imshow(beta)
    plt.title("Termination Probability")
    plt.colorbar()
    plt.show()

def visualizePolicy(option):
    pi = []
    for i in range(13):
        row = []
        for j in range(13):
            s = agent.sIdx((i, j))
            row.append(np.argmax(option.pi(s).detach()))
        pi.append(row)
    pi = (np.array(pi) + 1) * (env.grid >= 0)
    plt.imshow(pi)
    plt.title("Argmax Policy (1=left, 2=up, 3=right, 4=down)")
    plt.colorbar()
    plt.show()

for o in agent.options:
    visualizeTermination(o)
    visualizePolicy(o)
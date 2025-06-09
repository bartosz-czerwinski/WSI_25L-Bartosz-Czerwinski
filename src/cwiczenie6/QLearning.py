import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, action_space, gamma=0.9, beta=0.1, epsilon=0.1):
        self.action_space = action_space
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        current = self.Q[state][action]
        target_next = 0 if done else np.max(self.Q[next_state])
        delta = reward + self.gamma * target_next - current

        self.Q[state][action] = current + self.beta * delta

    def train(self, env, e_max=1000, t_max=100):
        rewards = []
        n_states = env.observation_space.n
        for episode in range(e_max):

            # stan początkowy zawsze lewy dolny róg planszy
            # state, _ = env.reset()
            # done = False
            # t = 0

            # losowy stan początkowy
            state0 = np.random.randint(n_states)
            env.reset()
            env.unwrapped.s = state0
            state = state0
            done = False
            t = 0

            episode_reward = 0
            while not done and t < t_max:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                if done:
                    episode_reward += 1
                    reward = 100 # nagroda za dotarcie do celu


                self.update(state, action, reward, next_state, done)
                state = next_state
                t += 1
            rewards.append(episode_reward)
        return self.Q, rewards
from QLearning import QLearningAgent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

def visualize_path(moves, title='Map of optimal moves'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    for i in range(4):
        for j in range(12):
            s = i * 12 + j
            if i == 3 and j == 11:
                label = 'G'
            elif i == 3 and 1 <= j <= 10:
                label = '0'
            else:
                label = action_arrows[moves[s]]

            ax.text(
                j, 3 - i, label,
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='square', facecolor='white', edgecolor='black')
            )

    ax.set_title(title)
    plt.savefig(title)
    plt.close()


def gamma_values_test(gamma_values, e_max=1000, t_max=20, repeats=20):
    step = 50
    env = gym.make('CliffWalking-v0')
    for gamma in gamma_values:
        avg_rewards = []
        for i in range(repeats):
            print(f"Training with gamma={gamma}, iteration {i + 1}/{repeats}...")
            agent = QLearningAgent(env.action_space, gamma=gamma, beta=0.9, epsilon=0.1)
            _, curr_rewards = agent.train(env, e_max=e_max, t_max=t_max)
            avg_rewards.append(curr_rewards)
        avg_rewards = np.mean(avg_rewards, axis=0)

        x = np.arange(1, e_max + 1, step)
        n = e_max // step
        y = [
            avg_rewards[g * step: (g + 1) * step].mean()
            for g in range(n)
        ]

        plt.plot(x, y, label=f'{gamma}')

    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    plt.title('Wpływ parametru gamma na wyniki')
    plt.legend()
    plt.grid()
    plt.savefig('gamma_test.png', dpi=600)
    plt.close()


def beta_values_test(beta_values, e_max=1000, t_max=20, repeats=20):
    step = 50
    env = gym.make('CliffWalking-v0')
    for beta in beta_values:
        avg_rewards = []
        for i in range(repeats):
            print(f"Training with beta={beta}, iteration {i + 1}/{repeats}...")
            agent = QLearningAgent(env.action_space, gamma=0.9, beta=beta, epsilon=0.1)
            _, curr_rewards = agent.train(env, e_max=e_max, t_max=t_max)
            avg_rewards.append(curr_rewards)
        avg_rewards = np.mean(avg_rewards, axis=0)
        x = np.arange(1, e_max + 1, step)
        n = e_max // step
        y = [
            avg_rewards[g * step: (g + 1) * step].mean()
            for g in range(n)
        ]

        plt.plot(x, y, label=f'{beta}')

    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    plt.title('Wpływ parametru beta na wyniki')
    plt.legend()
    plt.grid()
    plt.savefig('beta_test.png', dpi=600)
    plt.close()


def epsilon_values_test(epsilon_values, e_max=1000, t_max=20, repeats=20):
    step = 50
    env = gym.make('CliffWalking-v0')
    for epsilon in epsilon_values:
        avg_rewards = []
        for i in range(repeats):
            print(f"Training with epsilon={epsilon}, iteration {i + 1}/{repeats}...")
            agent = QLearningAgent(env.action_space, gamma=0.9, beta=0.1, epsilon=epsilon)
            _, curr_rewards = agent.train(env, e_max=e_max, t_max=t_max)
            avg_rewards.append(curr_rewards)
        avg_rewards = np.mean(avg_rewards, axis=0)
        x = np.arange(1, e_max + 1, step)
        n = e_max // step
        y = [
            avg_rewards[g * step: (g + 1) * step].mean()
            for g in range(n)
        ]

        plt.plot(x, y, label=f'{epsilon}')

    plt.xlabel('Epizod')
    plt.ylabel('Wynik')
    plt.title('Wpływ parametru epsilon na wyniki')
    plt.legend()
    plt.grid()
    plt.savefig('epsilon_test.png', dpi=600)
    plt.close()


# gamma_values_test([0.01, 0.1, 0.5, 0.9])
# beta_values_test([0.0001, 0.001, 0.01, 0.1, 0.5, 0.9], e_max=10000)
# epsilon_values_test([0.9, 0.5, 0.2, 0.1, 0.001, 0.0001, 0])



env = gym.make('CliffWalking-v0')
q_agent = QLearningAgent(env.action_space, gamma=0.9, beta=0.9, epsilon=0.001)
Q_q, _ = q_agent.train(env, e_max=100000, t_max=100)

moves_q = np.array([int(np.argmax(Q_q[s])) for s in range(env.observation_space.n)])
visualize_path(moves_q, title='Nauczona mapa ruchów')

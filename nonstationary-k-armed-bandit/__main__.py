import sys

import numpy as np

def argmaxes(array):
    max = array[0]
    results = []
    for i, e in enumerate(array):
        if e > max:
            max = e
            results.clear()
            results.append(i)
        elif e == max:
            results.append(i)
    return np.array(results)

class Bandit(object):
    """
    Gaussian random-walk Bandit

    The bandit arm q* value function changes for every step by a noise value
    defined as eta ~ N(0, 0.01^2)
    """
    def __init__(self, num_arms, walk_mean=0, walk_std_dev=0.01, reward_std_dev=1):
        self.num_arms = num_arms
        self.walk_mean = walk_mean
        self.walk_std_dev = walk_std_dev
        self.reward_std_dev = reward_std_dev
        self.means = np.zeros(num_arms)

    def update(self):
        self.means += np.random.normal(self.walk_mean, self.walk_std_dev, len(self.means))

    def pull(self, arm):
        return np.random.normal(self.means[arm], self.reward_std_dev)

class EpsilonGreedyAgent:
    """
    Epsilon-greedy RL Agent

    Uses value-function update q_{n+1}(a) = q_n(a) + a(n)(q_n(a) - r_n)
    where a(n) is passed as a parameter
    """
    def __init__(self, num_actions, a_n, epsilon=0.1):
        self.a_n = a_n
        self.epsilon = epsilon
        self.q_n = np.zeros(num_actions)
        self.action = None
        self.action_counter = np.zeros(num_actions)
    
    def policy(self):
        if np.random.random() < self.epsilon:
            action = np.random.choice(len(self.q_n))

        else:
            greedy_action = argmaxes(self.q_n)

            # break ties randomly
            if greedy_action.size == 1:
                action = greedy_action
            else:
                action = np.random.choice(greedy_action)
        
        self.action = action
        self.action_counter[action] += 1
        return action
    
    def update(self, reward):
        self.q_n[self.action] += self.a_n(self.action_counter[self.action])*(reward - self.q_n[self.action])

class Simulator:
    """
    Environment simulator
    Runs the agent, keeps track of score, updates the bandit
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
    
    def run(self, num_runs, num_steps):
        rewards = np.zeros((self.num_arms, num_steps))
        optimal_ratios = np.zeros((self.num_arms, num_steps))

        for i in range(num_runs):
            bandit = Bandit(self.num_arms)
            sample_average_agent = EpsilonGreedyAgent(self.num_arms, lambda n: 1/float(n))
            constant_step_agent = EpsilonGreedyAgent(self.num_arms, lambda n: 0.1)
            agents = [sample_average_agent, constant_step_agent]

            self.run_epoch(bandit, agents, num_runs, num_steps, rewards, optimal_ratios)

        rewards = rewards / float(num_runs)
        optimal_ratios = optimal_ratios / float(num_runs)

        return rewards, optimal_ratios

    def run_epoch(self, bandit, agents, num_runs, num_steps, rewards, optimal_ratios):
        for step in range(num_steps):
            for agent_i, agent in enumerate(agents):
                action = agent.policy()
                reward = bandit.pull(action)
                agent.update(reward)

                rewards[agent_i][step] += reward
                if bandit.means[action] == np.max(bandit.means):
                    optimal_ratios[agent_i][step] += 1
            
            bandit.update()

if __name__ == "__main__":
    k_arms = 10
    num_steps = 10000
    num_runs = 300

    filename = sys.argv[1]

    simulator = Simulator(k_arms)
    rewards, optimal_actions = simulator.run(num_runs, num_steps)

    np.savetxt(filename, (rewards[0], optimal_actions[0], rewards[1], optimal_actions[1]))

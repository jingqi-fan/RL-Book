import numpy as np
import matplotlib.pyplot as plt


### EPSILON-GREEDY ALGORITHM ###
# Given unknown horizon and unknown gaps between arms

class EpsilonGreedy(object):
    def __init__(self, rew_avg, epsilon):  ## Initialization
        self.means = rew_avg                     # vector of true means of the arms
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(self.means)   # best arm given the true mean rewards
        self.chosen_arm = int                    # arm chosen for exploitation
        self.num_pulls = np.zeros(rew_avg.size)  # vector of number of times that arm k has been pulled
        self.emp_means = np.zeros(rew_avg.size)  # vector of empirical means of arms
        self.cum_reg = [0]                       # cumulative regret
        self.time = 0.0
        self.epsilon = epsilon                   # exploration probability

        self.restart()


    def restart(self):  ## Restart the algorithm
        self.time = 0.0
        self.num_pulls = np.zeros(self.num_arms)
        self.emp_means = np.zeros(self.num_arms)
        self.cum_reg = [0]


    def get_best_arm(self):  ## Choose arm based on epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            self.chosen_arm = np.random.randint(self.num_arms)  # Explore
        else:
            self.chosen_arm = np.argmax(self.emp_means)  # Exploit

    def update_stats(self, arm, rew):  ## Update empirical means and number of pulls
        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1) + rew[arm]) / self.num_pulls[arm]
        self.time += 1


    def update_reg(self, arm, rew_vec):  ## Update the cumulative regret vector
        reg = rew_vec[self.genie_arm] - rew_vec[arm]  # regret as the "loss" in reward
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)


    def iterate(self, rew_vec):  ## Iterate the algorithm
        self.get_best_arm()
        self.update_reg(self.chosen_arm, rew_vec)
        self.update_stats(self.chosen_arm, rew_vec)



### BANDIT ARM REWARD NOISE FUNCTION ###

def get_reward(rew_avg):
    # Generate Bernoulli rewards for each arm
    reward = np.random.binomial(1, rew_avg)
    return reward


### DRIVER ALGO ###

def run_algo(rew_avg, num_iter, num_trial, epsilon):
    regret = np.zeros((num_trial, num_iter))
    algo = EpsilonGreedy(rew_avg, epsilon)

    for k in range(num_trial):
        algo.restart()

        if (k + 1) % 10 == 0:
            print('Instance number = ', k + 1)

        for t in range(num_iter - 1):
            rew_vec = get_reward(rew_avg)
            algo.iterate(rew_vec)
        regret[k, :] = np.asarray(algo.cum_reg)

    return regret


if __name__ == '__main__':
    ### INITIALIZE EXPERIMENT PARAMETERS ###
    # rew_avg = np.asarray([0.8, 0.96, 0.7, 0.5, 0.4, 0.3])
    rew_avg = np.linspace(0.1, 0.9, num=10)
    num_iter, num_trial, epsilon = int(5e4), 1, 0.1

    reg = run_algo(rew_avg, num_iter, num_trial, epsilon)
    avg_reg = np.mean(reg, axis=0)
    avg_reg.shape

    ### PLOT RESULT ###
    plt.plot(avg_reg, label="Epsilon-Greedy Avg. Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of Epsilon-Greedy Bandit')
    plt.legend()
    plt.show()

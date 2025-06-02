import numpy as np
import matplotlib.pyplot as plt


### THOMPSON SAMPLING ALGORITHM ###
# Given unknown horizon and unknown gaps between arms

class ThompsonSampling(object):
    def __init__(self, rew_avg):  ## Initialization
        self.means = rew_avg                     # vector of true means of the arms
        self.num_arms = rew_avg.size             # number of arms (k)
        self.genie_arm = np.argmax(self.means)   # best arm given the true mean rewards
        self.chosen_arm = int                    # arm chosen for exploitation
        self.alpha = np.ones(rew_avg.size)       # vector of alpha parameters (successes)
        self.beta = np.ones(rew_avg.size)        # vector of beta parameters (failures)
        self.cum_reg = [0]                       # cumulative regret
        self.time = 0.0

        self.restart()


    def restart(self):  ## Restart the algorithm
        self.time = 0.0
        self.alpha = np.ones(self.num_arms)
        self.beta = np.ones(self.num_arms)
        self.cum_reg = [0]


    def get_best_arm(self):  ## For each time index, find the best arm according to Thompson Sampling
        samples = np.random.beta(self.alpha, self.beta)
        self.chosen_arm = np.argmax(samples)

    def update_stats(self, arm, rew):  ## Update the success and failure counts
        self.alpha[arm] += rew[arm]
        self.beta[arm] += 1 - rew[arm]
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

def run_algo(rew_avg, num_iter, num_trial):
    regret = np.zeros((num_trial, num_iter))
    algo = ThompsonSampling(rew_avg)

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
    num_iter, num_trial = int(5e4), 1

    reg = run_algo(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)
    avg_reg.shape

    ### PLOT RESULT ###
    plt.plot(avg_reg, label="Thompson Sampling Avg. Regret")
    plt.xlabel('iterations')
    plt.ylabel('cumulative regret')
    plt.title('Cumulative Regret of Thompson Sampling Bandit')
    plt.legend()
    plt.show()

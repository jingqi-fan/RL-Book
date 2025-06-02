import numpy as np
import matplotlib.pyplot as plt


class UCB():
    def __init__(self, rew_avg, num_iter):
        self.means = rew_avg  # 真实均值
        self.num_iter = num_iter  # 最大时间步
        self.num_arms = rew_avg.size  # 总臂数
        self.genie_arm = np.argmax(rew_avg)  # 最优臂
        self.num_pulls = np.zeros(self.num_arms)  # 每个臂被拉的次数
        self.emp_means = np.zeros(self.num_arms)  # 经验均值
        self.ucb_values = np.full(self.num_arms, np.inf)  # UCB 上界值
        self.cum_reg = [0]  # 累积懊悔

    def get_best_arm(self):
        """根据 UCB 选择当前最优臂"""
        return int(np.argmax(self.ucb_values))

    def update_stats(self, rew, arm, t):
        """更新经验均值和 UCB 置信上界"""
        self.num_pulls[arm] += 1
        self.emp_means[arm] = (self.emp_means[arm] * (self.num_pulls[arm] - 1) + rew[arm]) / self.num_pulls[arm]
        self.ucb_values[arm] = self.emp_means[arm] + np.sqrt(2 * np.log(t + 1) / self.num_pulls[arm])

    def update_reg(self, rew_vec, arm):
        """更新累积懊悔"""
        reg = rew_vec[self.genie_arm] - rew_vec[arm]
        reg += self.cum_reg[-1]
        self.cum_reg.append(reg)

    def iterate(self, rew_vec, t):
        """运行 UCB 迭代步骤"""
        arm = self.get_best_arm()
        self.update_stats(rew_vec, arm, t)
        self.update_reg(rew_vec, arm)


def get_reward(rew_avg):
    """模拟带有高斯噪声的奖励"""
    mean = np.zeros(rew_avg.size)
    cov = 0.01 * np.eye(rew_avg.size)
    noise = np.random.multivariate_normal(mean, cov)
    reward = rew_avg + noise
    return reward


def run_algo(rew_avg, num_iter, num_trial):
    regret = np.zeros((num_trial, num_iter))
    for k in range(num_trial):
        algo = UCB(rew_avg, num_iter)
        algo.cum_reg = [0]

        if (k + 1) % 10 == 0:
            print(f'Instance {k + 1}')

        for t in range(num_iter):
            rew_vec = get_reward(rew_avg)
            algo.iterate(rew_vec, t)

        # regret[k, :] = np.asarray(algo.cum_reg)
        regret[k, :] = np.asarray(algo.cum_reg[:num_iter])  # 只取前 num_iter 个数据

    return regret


if __name__ == '__main__':
    rew_avg = np.linspace(0.1, 0.9, num=10)  # 设定臂的真实均值
    num_iter, num_trial = 1500, 1  # 设定迭代次数和实验次数

    reg = run_algo(rew_avg, num_iter, num_trial)
    avg_reg = np.mean(reg, axis=0)

    # 绘制累积懊悔曲线
    plt.plot(avg_reg, label="UCB")
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with UCB')
    plt.legend()
    plt.show()

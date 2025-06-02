import numpy as np
import matplotlib.pyplot as plt
import os
from ucb1 import run_algo as run_ucb
from ts import run_algo as run_ts
from epsilon_greedy import run_algo as run_epsilon_greedy
from elimination import run_algo as run_se
from etc import run_algo as run_etc

# 设置字体为 Times New Roman
# 启用 LaTeX 渲染
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为 Times New Roman
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # 加载 amsmath 包

# 设置字体大小
plt.rcParams['axes.labelsize'] = 26  # 设置轴标签字体大小
plt.rcParams['legend.fontsize'] = 20  # 设置图例字体大小
plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 20  # 设置y轴刻度字体大小

### TEST AND COMPARE POLICIES ###
if __name__ == '__main__':
    ### INITIALIZE EXPERIMENT PARAMETERS ###
    rew_avg = np.linspace(0.1, 0.9, num=6)
    num_iter, num_trial = 1000, 1
    epsilon = 0.01  # Exploration probability for epsilon-greedy
    m_etc = 100  # Explore-exploit threshold for ETC
    npy_file = 'regret_data.npy'

    if os.path.exists(npy_file):
        print("Loading precomputed regret data...")
        data = np.load(npy_file, allow_pickle=True).item()
        avg_reg_ucb = data['UCB']
        avg_reg_ts = data['Thompson Sampling']
        avg_reg_eps = data['Epsilon-Greedy']
        avg_reg_se = data['Successive Elimination']
        avg_reg_etc = data['ETC']
    else:
        print("Running experiments and saving results...")
        reg_ucb = run_ucb(rew_avg, num_iter, num_trial)
        avg_reg_ucb = np.mean(reg_ucb, axis=0)

        reg_ts = run_ts(rew_avg, num_iter, num_trial)
        avg_reg_ts = np.mean(reg_ts, axis=0)

        reg_eps = run_epsilon_greedy(rew_avg, num_iter, num_trial, epsilon)
        avg_reg_eps = np.mean(reg_eps, axis=0)

        reg_se = run_se(rew_avg, num_iter, num_trial)
        avg_reg_se = np.mean(reg_se, axis=0)

        reg_etc = run_etc(rew_avg, m_etc, num_iter, num_trial)
        avg_reg_etc = np.mean(reg_etc, axis=0)

        # data = {
        #     'UCB': avg_reg_ucb,
        #     'Thompson Sampling': avg_reg_ts,
        #     'Epsilon-Greedy': avg_reg_eps,
        #     'Successive Elimination': avg_reg_se,
        #     'ETC': avg_reg_etc
        # }
        # np.save(npy_file, data)

    ### PLOT RESULTS ###
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iter), avg_reg_ucb, label='UCB', color='blue')
    plt.plot(range(num_iter), avg_reg_ts, label='TS', color='green')
    plt.plot(range(num_iter), avg_reg_eps, label='e-Greedy', color='red')
    plt.plot(range(num_iter), avg_reg_se, label='SE', color='purple')
    plt.plot(range(num_iter), avg_reg_etc, label='ETC', color='orange')

    plt.xlabel('$t$')
    plt.ylabel('$R_t$')
    # plt.title('Comparison of Bandit Policies: Cumulative Regret Growth')
    plt.legend()
    plt.grid()
    plt.ylim(0, 200)

    # 保存为 PDF
    plt.savefig("result.pdf", format="pdf", bbox_inches="tight")  # 确保边界紧凑

    plt.show()

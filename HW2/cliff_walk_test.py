import numpy as np
import matplotlib.pyplot as plt
QL_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/reward_zero.npy')
SA_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/SARSA/reward_zero.npy')

plt.figure(1)
plt.plot(QL_nozero)
plt.plot(SA_nozero)
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/figures/Cliff_nozero.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# QL_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/QLearning/reward_zero.npy')
# SA_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/SARSA/reward_zero.npy')

# plt.figure(1)
# plt.plot(QL_nozero)
# plt.plot(SA_nozero)
# plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/figures/Cliff_nozero.png')
# plt.show()

# 将状态变为横纵坐标, set status into an one-axis coordinate value
def _state_to_xy(s):
    x = s % 12
    y = int((s - x) / 12)
    return x,y

width = 12
height = 4

Q_QL = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/SARSA/Q_Table_nozero.npy')
Map = np.zeros((height, width))

for k in range(48):
    j, i = _state_to_xy(k)
    Map[height - i - 1, j] = np.argmax(Q_QL[k, :])

plt.imshow(Map, data='True')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
QL_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_QL/reward_nozero.npy')
SA_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_SA/reward_nozero.npy')
DynaQ_nozero = np.load('/Users/xiaojian_xiang/Projects/AI3606/HW2/T2_DynaQ/reward_nozero.npy')

plt.figure(1)
plt.plot(QL_nozero)
plt.plot(SA_nozero)
plt.plot(DynaQ_nozero)
plt.savefig('/Users/xiaojian_xiang/Projects/AI3606/HW2/figures/SO_nozero.png')
plt.show()

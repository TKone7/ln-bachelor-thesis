import matplotlib.pyplot as plt
import numpy as np
def calc_random():
    spread = np.random.rand(50) * (0.3) + 0.4
    center = np.ones(50) * 0.5
    flier_high = np.random.rand(10) * (1-0.75) + 0.75
    flier_low = np.random.rand(10) * 0.1
    data = np.concatenate((spread, center, flier_high, flier_low))
    return data

data = [calc_random(), calc_random(), calc_random()]

fig1, ax1 = plt.subplots()


ax1.set_title('Deistribution of Gini coefficients')
ax1.boxplot(data)
ax1.set_xticklabels(['10%', '50%', '90%'])

plt.savefig('distribution_gini.png', format='png')
plt.show()
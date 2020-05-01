import matplotlib.pyplot as plt

x = [x for x in range(1,110000)]
y = [0.5 - (0.5 * (i/len(x))) for i, c in enumerate(x)]

y2 = [y*0.8 for i, y in enumerate(y)]
y3 = [y*0.9 for i, y in enumerate(y2)]

plt.plot(x, y, color='blue', linestyle='solid', linewidth=2, markersize=6, label='10% participation')
plt.plot(x, y2, color='red', linestyle='solid', linewidth=2, markersize=6, label='50% participation')
plt.plot(x, y3, color='green', linestyle='solid', linewidth=2, markersize=6, label='90% participation')
plt.legend(loc='upper right')

plt.xlabel('nr of rebalancing operations')
plt.ylabel('Gini coefficient')

plt.savefig('rebal_op.png', format='png')
plt.show()
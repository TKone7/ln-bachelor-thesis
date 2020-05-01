import matplotlib.pyplot as plt
x = [x*10 for x in range(1,11)]
y = [10/c for c in x]

y2 = [y*0.1*(10-i) for i, y in enumerate(y)]
y3 = [y*0.1*(10-i) for i, y in enumerate(y2)]

plt.plot(x, y, color='blue', marker='o', linestyle='solid', linewidth=2, markersize=6, label='normal')
plt.plot(x, y2, color='red', marker='o', linestyle='solid', linewidth=2, markersize=6)
plt.plot(x, y3, color='green', marker='o', linestyle='solid', linewidth=2, markersize=6)
plt.legend(loc='upper right')

plt.xlabel('level of participation')
plt.ylabel('Gini coefficient')

plt.savefig('participation.png', format='png')
plt.show()
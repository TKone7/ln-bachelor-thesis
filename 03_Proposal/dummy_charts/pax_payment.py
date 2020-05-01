import matplotlib.pyplot as plt
import math
x = [x for x in range(1,10000)]
y = [math.log(c) /9 for c in x]

y2 = [y*0.9 for i, y in enumerate(y)]
y3 = [y*0.8 for i, y in enumerate(y2)]

plt.plot(x, y, color='blue', linestyle='solid', linewidth=2,  label='90% participation')
plt.plot(x, y2, color='red', linestyle='solid', linewidth=2, label='50% participation')
plt.plot(x, y3, color='green', linestyle='solid', linewidth=2, label='10% participation')
plt.legend(loc='upper right')

plt.xlabel('Max possible amount')
plt.ylabel('% of shortest paths')

plt.savefig('max_payable.png', format='png')
plt.show()
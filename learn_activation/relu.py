import numpy as np
from matplotlib import pyplot as plt

# 입력이 0을 넘으면 그 입력 출력, 아니면 0을 출력
def relu(x):
    return np.maximum(0, x)


x = np.arange(-20.0, 20.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()

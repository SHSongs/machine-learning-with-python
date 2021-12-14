import numpy as np
import matplotlib.pylab as plt

# 출력값이 0과 1 사이, 매끄러운 곡선으로 역전파 시, 기울기 폭주 발생 안함
# 분류는 0과 1 출력값이 어디에 가까운지에 따라 정해짐

# 입력값이 커도 0과 1 사이이기에 역전파시 기울기 0에 수렴하는 기울기 소실 발생
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-20.0, 20.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

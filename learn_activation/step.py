# 활성화 함수 Activation function
# 입력 신호의 총합이 활성화를 일으킬지 정함

import numpy as np
import matplotlib.pylab as plt

# X가 0.1 이든 10.0 이든 1로 반환한다.
# 강도에 대한 의미 부여 없음.
def step_function(x):
    # # 실수 안됨
    # if x > 0:
    #     return 1
    # else:
    #     return 0
    y = x > 0
    return y.astype(int)


x = np.array([-1.0, 1.0, 2.0])
y = step_function(x)
print(y)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()



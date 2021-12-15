import numpy as np


def softmax_1(a):
    sum_exp_a = np.sum(a)
    return a / sum_exp_a


def softmax_2(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


# softmax 함수 개선
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


a = np.array([0.1, 0.9])
y = softmax_1(a)
print(y)

a = np.array([0.1, 0.9])
y = softmax_2(a)
print(y)

a = np.array([0.1, 0.9])

y = softmax(a)
print(y)

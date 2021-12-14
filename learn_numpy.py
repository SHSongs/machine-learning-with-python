import numpy as np


def identity_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 2 * 3 행렬
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = identity_function(a1)
    print(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = identity_function(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


def for_forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    x_h = x.shape[0]

    W1_h, W1_w = W1.shape
    a1 = np.zeros(shape=W1_w)
    for i in range(W1_w):  # 3
        for j in range(W1_h):  # 2
            a1[i] += (W1[j][i] * x[j])

    b1_h = b1.shape[0]
    for i in range(b1_h):
        a1[i] += b1[i]

    print(a1)


    return a1


network = init_network()
x = np.array([1.0, 5.0])
y = forward(network, x)
# print(y)

y2 = for_forward(network, x)
# print(y2)

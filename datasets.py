import numpy as np

clf_sin = lambda X, K: X[:, 1] < np.sin(X[:, 0] * K)
clf_circle = lambda X, K: np.sqrt(X[:, 1]**2 + X[:, 0]**2) < K
def clf_spirals(X, K):
    r = np.sqrt(X[:, 1]**2 + X[:, 0]**2) + 1e-10
    phi = np.arccos(X[:, 1] / r)
    phi[X[:, 0] < 0] = -phi[X[:, 0] < 0]
    return np.sin(phi + r * K) > 0

reg_sin = lambda X, K: np.sin(X[:, 0] * K)
reg_tanh = lambda X, K: 1 / (1 + np.exp(-X[:, 0] * K)) * 2 - 0.5
reg_steps = lambda X, K: np.sign(reg_sin(X, K))
reg_relu = lambda X, K: (X > 0) * X * K

dataset_functions = {
    'Classification': {
        'Sin': clf_sin,
        'Circle': clf_circle,
        'Spirals': clf_spirals,
    },
    'Regression': {
        'Sin': reg_sin,
        'Sigmoid': reg_tanh,
        'Steps': reg_steps,
        'ReLU': reg_relu,
    },
}

import numpy as np


def dot(A, B):
    # A: MxN
    # B: NxP

    M, N = A.shape
    N2, P = B.shape

    if N != N2:
        raise ValueError("Dimensiones incompatibles para multiplicación de matrices")

    C = np.zeros((M, P), dtype=np.float32)

    for i in range(M):
        for j in range(P):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    return C


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Linear:
    def __init__(self, in_, out_, rng, is_sec=False):
        self.input = None
        self.output = None

        # np.dot internamente es paralela.
        self.prod_cruz = np.dot if not is_sec else dot

        # kelming-he
        self.weights = rng.standard_normal((in_, out_)) * np.sqrt(2. / in_)
        self.bias = np.zeros((1, out_)).astype(np.float32)

    def forward(self, x):
        self.input = x
        self.output = self.prod_cruz(x, self.weights) + self.bias

        return self.output

    def backward(self, z, lr=0.001):
        grad_input = self.prod_cruz(z, self.weights.T)  # dL/dx = dL/dz * dz/dx = z * w
        grad_weights = self.prod_cruz(self.input.T, z)  # dL/dw = dL/dz * dz/dw = z * x
        grad_bias = np.sum(z, axis=0, keepdims=True)  # dL/db = dL/dz * dz/db = 1

        # Usando el optimizador SGD
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias

        return grad_input


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x

        xp = x.copy()
        xp[xp < 0] = 0
        return xp

    def backward(self, z):
        grad = np.where(self.input > 0, z, 0)
        return grad


class FullyConnected:

    def __init__(self, in_, out_, rng, num_capas=5, is_sec=False):

        self.num_capas = num_capas
        middle = (num_capas - 2) // 2

        lg2 = np.log2(in_)
        lower = int(2 ** np.floor(lg2))
        upper = int(2 ** np.ceil(lg2))

        # crece demasiado pipi
        #near = lower if abs(in_ - lower) < abs(in_ - upper) else upper
        near = 256 # buena cantidad de inicio para digitos mnist.
        factor = lambda val, i: val * 2 if middle > i else val // 2

        self.capa_0 = Linear(in_, near, rng, is_sec=is_sec)
        self.relu_0 = ReLU()

        for i in range(1, num_capas - 1):
            past_increm = int(near)
            near = int(factor(near, i-1))

            #print(past_increm, near)

            capa_i = Linear(past_increm, near, rng, is_sec=is_sec)
            relu_i = ReLU()
            setattr(self, f"capa_{i}", capa_i)
            setattr(self, f"relu_{i}", relu_i)

        self.capa_f = Linear(near, out_, rng, is_sec=is_sec)

    def forward(self, x):

        if not isinstance(x, np.ndarray):
            x = np.array(x).astype(np.float32)

        for i in range(self.num_capas - 1):
            capa_i = getattr(self, f"capa_{i}")
            relu_i = getattr(self, f"relu_{i}")
            x = capa_i.forward(x)
            x = relu_i.forward(x)

        last_capa = self.capa_f.forward(x)

        return softmax(last_capa)

    def backward(self, z, lr=0.001):

        grad = self.capa_f.backward(z=z, lr=lr)

        for i in range(self.num_capas - 2, -1, -1):
            relu_i = getattr(self, f"relu_{i}")
            capa_i = getattr(self, f"capa_{i}")

            grad = relu_i.backward(z=grad)
            grad = capa_i.backward(z=grad, lr=lr)

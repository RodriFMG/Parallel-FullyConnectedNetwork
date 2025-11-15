import numpy as np


def to_batch(data, label, batch_size=64, shuffle=True, rng=None):
    if not isinstance(data, np.ndarray) or not np.issubdtype(data.dtype, np.float32):
        data = np.array(data).astype(np.float32)
    if not isinstance(label, np.ndarray) or not np.issubdtype(label.dtype, np.int64):
        label = np.array(label).astype(np.int64)

    if data.shape[0] != label.shape[0]:
        raise ValueError(f"Cantidad dispareja de muestras y labels: \n"
                         f"Data: {data.shape[0]}\n"
                         f"Labels: {label.shape[0]}")

    total_data = data.shape[0]

    if shuffle:

        if rng is None:
            raise ValueError("Si shuffle=True se debe mandar un generador independiente")

        indexs = rng.permutation(total_data)
        data = data[indexs]
        label = label[indexs]

    data = data.reshape(data.shape[0], -1)

    return [
        [data[step:step + batch_size], label[step:step + batch_size]]
        for step in range(0, total_data, batch_size)
    ]


def LossFunction(z, label):
    N = label.shape[0]
    pred = z[np.arange(N), label]
    return -1 / N * np.sum(np.log(pred + 1e-8))


def GradLossFunction(z, label):
    N = label.shape[0]
    grad = z.copy()
    grad[np.arange(N), label] -= 1

    # Como el gradiante lleva -1/N, arrastra ese 1/N al derivar...
    grad /= N
    return grad


def accuracy(model, eval_data):
    N = 0
    Corrects = 0

    for data, label in eval_data:

        z = model.forward(data)
        pred = np.argmax(z, axis=1)
        Corrects += np.sum(pred == label)
        N += label.shape[0]

    return float(Corrects) / N


# Se supone que train_data ya debe estar batcherizado (xd)
def train(model, train_data, eval_data, epochs=100, show=10, lr=0.001):
    avg_loss = None
    factor_loss = 0.1
    for epoch in range(epochs):
        for data, label in train_data:
            z = model.forward(data)
            loss = LossFunction(z, label=label)
            z_grad = GradLossFunction(z, label=label)
            model.backward(z_grad, lr=lr)

            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = factor_loss * loss + (1 - factor_loss) * avg_loss

        if (epoch + 1) % show == 0:

            acc = accuracy(model, eval_data)
            print(f"Epoca: {epoch+1} --> avg_loss: {avg_loss:.5f}, accuracy: {acc:.5f}")

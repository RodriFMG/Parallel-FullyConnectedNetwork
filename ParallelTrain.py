from joblib import Parallel, delayed
import joblib
import numpy as np
from DataToArray import get_images
from FCN.method import train, accuracy, to_batch, split_data
from FCN.model import FullyConnected


def SecTrain(seed, SampleData, Labels, is_sec=False, BatchSize=64, Epochs=10, NumLayers=5, lr=0.01):
    SeedRng = np.random.default_rng(seed=seed)

    train_d, eval_d, _ = split_data(SampleData, Labels, [0.85, 0.10, 0.05])
    DataBatchTrain = to_batch(train_d[0], train_d[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)
    DataBatchEval = to_batch(eval_d[0], eval_d[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)

    model = FullyConnected(in_=28 * 28, out_=10, num_capas=NumLayers, rng=SeedRng, is_sec=is_sec)
    train(model, DataBatchTrain, DataBatchEval, epochs=Epochs, lr=lr)

    # Luego lo pongo con test data
    A_s = accuracy(model, eval_data=DataBatchEval)
    return [model, A_s]


def parallel_train(DataPath, is_sec=False, num_iters=10, percentage=1, NumThreads=5, BatchSize=64, Epochs=10,
                   NumLayers=5, lr=0.01):
    if not 0 <= percentage <= 1:
        raise ValueError("El % de data a usar debe estar entre [0, 1]")

    SampleData, Labels = get_images(DataPath, percentage)

    p = num_iters if NumThreads == -1 else NumThreads

    results = Parallel(n_jobs=p)(
        delayed(SecTrain)(seed=i, SampleData=SampleData, Labels=Labels, is_sec=is_sec,
                          BatchSize=BatchSize, Epochs=Epochs, NumLayers=NumLayers, lr=lr)
        for i in range(num_iters)
    )

    best_model, A_s = max(results, key=lambda x: x[1])

    return best_model, A_s

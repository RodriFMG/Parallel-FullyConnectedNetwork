from joblib import Parallel, delayed
import joblib
import numpy as np
from DataToArray import get_images
from FCN.method import train, accuracy, to_batch, split_data
from FCN.model import FullyConnected


def SecTrain(seed, SampleData, Labels, BatchSize=64, Epochs=10, NumLayers=5, lr=0.01):
    SeedRng = np.random.default_rng(seed=seed)

    train_d, eval_d, _ = split_data(SampleData, Labels, [0.85, 0.10, 0.05])
    DataBatchTrain = to_batch(train_d[0], train_d[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)
    DataBatchEval = to_batch(eval_d[0], eval_d[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)

    model = FullyConnected(in_=28 * 28, out_=10, num_capas=NumLayers, rng=SeedRng)
    train(model, DataBatchTrain, DataBatchEval, epochs=Epochs, lr=lr)

    # Luego lo pongo con test data
    A_s = accuracy(model, eval_data=DataBatchEval)
    return [model, A_s]


def parallel_train(DataPath, NumThreads=5, BatchSize=64, Epochs=10, NumLayers=5, lr=0.01):

    SampleData, Labels = get_images(DataPath)

    # Un hilo por modelo
    results = Parallel(n_jobs=NumThreads)(
        delayed(SecTrain)(seed=i, SampleData=SampleData, Labels=Labels,
                          BatchSize=BatchSize, Epochs=Epochs, NumLayers=NumLayers, lr=lr)
        for i in range(NumThreads)
    )

    best_model, A_s = max(results, key=lambda x: x[1])
    joblib.dump(best_model, "fcn_model.joblib")

    return best_model, A_s

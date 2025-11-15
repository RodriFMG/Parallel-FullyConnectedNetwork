from joblib import Parallel, delayed
import joblib
import numpy as np
from DataToArray import get_images
from FCN.method import train, accuracy, to_batch
from FCN.model import FullyConnected


def SecTrain(seed, Data, BatchSize=64, Epochs=10, NumLayers=5, lr=0.01):
    SeedRng = np.random.default_rng(seed=seed)
    DataBatchTrain = to_batch(Data[0], Data[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)
    DataBatchEval = to_batch(Data[2], Data[3], batch_size=BatchSize, shuffle=True, rng=SeedRng)

    model = FullyConnected(in_=28 * 28, out_=10, num_capas=NumLayers, rng=SeedRng)
    train(model, DataBatchTrain, DataBatchEval, epochs=Epochs, lr=lr)

    # Luego lo pongo con test data
    A_s = accuracy(model, eval_data=DataBatchEval)
    return [model, A_s]


def parallel_train(DataPath, NumThreads=5, BatchSize=64, Epochs=10, NumLayers=5, lr=0.01):
    x_train, y_train, x_eval, y_eval = get_images(DataPath)
    Data = [x_train, y_train, x_eval, y_eval]

    # Un hilo por modelo
    results = Parallel(n_jobs=NumThreads)(
        delayed(SecTrain)(seed=i, Data=Data, BatchSize=BatchSize, Epochs=Epochs, NumLayers=NumLayers, lr=lr)
        for i in range(NumThreads)
    )

    best_model, _ = max(results, key=lambda x: x[1])
    joblib.dump(best_model, "fcn_model.joblib")

    return best_model

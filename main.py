from ParallelTrain import parallel_train

import joblib
from DataToArray import get_images
from FCN.method import accuracy, to_batch
import numpy as np


def get_best_parallel_model(DataPath="./Data/"):

    NumThreads = 10
    BatchSize = 64
    Epochs = 9
    NumLayers = 4
    lr = 0.001

    model = parallel_train(DataPath, NumThreads=NumThreads, BatchSize=BatchSize,
                           Epochs=Epochs, NumLayers=NumLayers, lr=lr)

    return model


def load_model(path="./fnc_model.joblib"):
    if not path.endswith(".joblib"):
        raise ValueError("Se esperaba un modelo con el contenedor joblib")
    model = joblib.load(path)

    return model


def test_load_model(DataPath="./Data/", path_load="./fcn_model.joblib"):
    BatchSize = 64

    _, _, x_eval, y_eval = get_images(DataPath)

    model = load_model(path_load)

    SeedRng = np.random.default_rng(seed=100)
    DataBatchEval = to_batch(x_eval, y_eval, batch_size=BatchSize, shuffle=True, rng=SeedRng)

    print(f"Accuracy: {accuracy(model, DataBatchEval)}")


if __name__ == "__main__":
    DataPath = "./Data/"
    path_load = "./fcn_model.joblib"

    test_load_model(DataPath, path_load)

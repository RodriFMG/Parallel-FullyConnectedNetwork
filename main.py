from ParallelTrain import parallel_train

import joblib
from DataToArray import get_images
from FCN.method import accuracy, to_batch, split_data
import numpy as np


def get_best_parallel_model(DataPath="./Data/"):

    NumThreads = 2
    BatchSize = 64
    Epochs = 9
    NumLayers = 4
    lr = 0.001

    model, A_s = parallel_train(DataPath, NumThreads=NumThreads, BatchSize=BatchSize,
                           Epochs=Epochs, NumLayers=NumLayers, lr=lr)

    print(f"Best accuracy: {A_s:.5f}")

    return model


def load_model(path="./fnc_model.joblib"):
    if not path.endswith(".joblib"):
        raise ValueError("Se esperaba un modelo con el contenedor joblib")
    model = joblib.load(path)

    return model


def test_load_model(DataPath="./Data/", path_load="./fcn_model.joblib"):
    BatchSize = 64

    SampleData, Labels = get_images(DataPath)
    _, _, test = split_data(SampleData, Labels, [0.85, 0.10, 0.05])

    model = load_model(path_load)

    SeedRng = np.random.default_rng(seed=100)
    DataBatchTest = to_batch(test[0], test[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)

    print(f"Accuracy: {accuracy(model, DataBatchTest)}")


if __name__ == "__main__":
    DataPath = "./Data/"
    path_load = "./fcn_model.joblib"

    test_load_model(DataPath, path_load)

from ParallelTrain import parallel_train
import matplotlib.pyplot as plt

import os
import joblib
from DataToArray import get_images
from FCN.method import accuracy, to_batch, split_data
import numpy as np
import time


def get_best_parallel_model(DataPath="./Data/", save=False):
    # Si NumThreads == -1 <- PRAM, numero de hilos = num_iters
    NumThreads = 1
    BatchSize = 64
    Epochs = 10
    NumLayers = 5
    lr = 0.001
    num_iters = 10
    percentage = 1
    is_sec = False

    start = time.perf_counter()
    model, A_s = parallel_train(DataPath, percentage=percentage, num_iters=num_iters, NumThreads=NumThreads,
                                BatchSize=BatchSize,
                                Epochs=Epochs, NumLayers=NumLayers, lr=lr, is_sec=is_sec)
    end = time.perf_counter()

    if save:
        joblib.dump(model, "fcn_model.joblib")

    print("Tiempo:", end - start, "segundos")
    print(f"Best accuracy: {A_s:.5f}")

    return model


def load_model(path="./fnc_model.joblib"):
    if not path.endswith(".joblib"):
        raise ValueError("Se esperaba un modelo con el contenedor joblib")
    model = joblib.load(path)

    return model


def test_load_model(DataPath="./Data/", path_load="./fcn_model.joblib"):
    BatchSize = 64

    SampleData, Labels = get_images(DataPath, percentage=0.1)
    _, _, test = split_data(SampleData, Labels, [0.85, 0.10, 0.05])

    model = load_model(path_load)

    SeedRng = np.random.default_rng(seed=100)
    DataBatchTest = to_batch(test[0], test[1], batch_size=BatchSize, shuffle=True, rng=SeedRng)

    print(f"Accuracy: {accuracy(model, DataBatchTest)}")


def test_model_times(DataPath, num_hilos, num_data, save=False):
    if num_data > 70_000:
        num_data = 70_000

    NumThreads = num_hilos
    BatchSize = 64
    Epochs = 15
    NumLayers = 5
    lr = 0.001
    percentage = num_data / 70_000
    num_iters = 25
    is_sec = False

    start = time.perf_counter()
    model, As = parallel_train(DataPath, percentage=percentage, NumThreads=NumThreads, BatchSize=BatchSize,
                               Epochs=Epochs, NumLayers=NumLayers, lr=lr, is_sec=is_sec, num_iters=num_iters)
    end = time.perf_counter()

    if save:
        print("/======================================/")
        print(f"Modelo con precision: {As}")
        print("/======================================/")

        joblib.dump(model, f"./model_np/fcn_model_{num_hilos}_{num_data}.joblib")

    return end - start


def test_times_np(DataPath, save=False):
    hilos = [1, 2, 4, 8, 16, 32]
    num_data = [5000, 10000, 20000, 40000]

    if save:
        os.makedirs("./model_np", exist_ok=True)

    times = [[test_model_times(DataPath, p, n, save=save) for n in num_data] for p in hilos]

    if save:
        np.save("./model_np/times_test.npy", np.array(times, dtype=object))

    plt.figure(figsize=(10, 6))

    for i, p in enumerate(hilos):
        plt.plot(num_data, times[i], marker='o', label=f"{p} hilos")

    plt.xlabel("Cantidad de datos usados en el entrenamiento")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Comparación de tiempos vs cantidad de datos para distintos números de hilos")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DataPath = "./Data/"
    path_load = "./fcn_model.joblib"

    test_times_np(DataPath, save=True)

from DataToArray import get_images
from FCN.method import train, accuracy, to_batch
from FCN.model import FullyConnected

if __name__ == "__main__":
    DataPath = "./Data/"
    x_train, y_train, x_eval, y_eval = get_images(DataPath)

    DataBatchTrain = to_batch(x_train, y_train, batch_size=64, shuffle=True)
    DataBatchEval = to_batch(x_eval, y_eval, batch_size=64, shuffle=True)

    model = FullyConnected(in_=28*28, out_=10, num_capas=5)
    train(model, DataBatchTrain, DataBatchEval)
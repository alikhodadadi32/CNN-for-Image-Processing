import argparse
import os

import tensorflow as tf

from DataLoader import load_data, train_valid_split, load_testing_images
from Model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument("--Resnet_version", default="V2", help="the version of ResNet")
parser.add_argument("--Num_residual_blocks", default=3, help="No. of blocks", type=int)
parser.add_argument(
    "--First_layer_num_filter", default=32, help="Firt layer filter", type=int
)
parser.add_argument("--Kernel_size", default=3, help="Kernel size", type=int)
parser.add_argument(
    "--List_residual_layers", help="No. of layers in each block", nargs="+", type=int
)
parser.add_argument("--batch_size", default=64, help="batch size", type=int)
parser.add_argument("--num_classes", default=10, help="No. of classses", type=int)
parser.add_argument("--save_interval", default=20, help="Save model at -- intervals")
parser.add_argument("--_lambda", default=0, help="ridge param.", type=float)
parser.add_argument("--_beta", default=2e-4, help="lasso param.", type=float)
parser.add_argument("--_alpha", default=0, help="elastic param.", type=float)
parser.add_argument("--modeldir", default="mode_v1", help="save directory")
parser.add_argument("--mymodel", default="Final_model", help="smodel no.")
parser.add_argument("--start_learning_rate", default=0.001, help="learning rate")
parser.add_argument("--mode", default="train", help="train, test or predict")
parser.add_argument(
    "--data_dir",
    default="C:/Users/akhod/Desktop/CSCE-636/Project/636_Project/data/cifar-10-python.tar.gz",
    help="path to the data",
)
parser.add_argument(
    "--save_dir",
    default="C:/Users/akhod/Desktop/CSCE-636/Project/636_Project/code",
    help="path to save the results",
)
parser.add_argument(
    "--test_data_dir",
    default="C:/Users/akhod/Desktop/CSCE-636/Project/636_Project/data/private_test_images.npy",
    help="path to the private test data",
)
args = parser.parse_args()


Configs = args


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sess = tf.Session()
    model = MyModel(sess, Configs)
    if args.mode == "train":
        x_train, y_train, x_test, y_test = load_data(args.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
        print(Configs)
        model.train(x_train, y_train, 200)

    elif args.mode == "test":
        _, _, x_test, y_test = load_data(args.data_dir)
        model.test_or_validate(x_test, y_test, [160])

    elif args.mode == "predict":
        x_test = load_testing_images(args.test_data_dir)
        model.predict(x_test, 160)

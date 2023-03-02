import sys
sys.path.append("..")
import argparse
from lib.dec import DEC
from lib.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEC MNIST Example')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='learning rate for training (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='update-interval  (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--pretrain', type=str, default="../model/sdae.pt", metavar='N',
                        help='use pre-trained weights')
    args = parser.parse_args()


    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels

    dec = DEC(input_dim=784, z_dim=10, n_clusters=10,
              encodeLayer=[500, 500, 2000], activation="relu", dropout=0)
    print(dec)
    dec.load_model(args.pretrain)
    dec.fit(X, y, lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
            update_interval=args.update_interval)


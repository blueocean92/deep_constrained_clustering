import sys
sys.path.append("..")
import torch.utils.data
import numpy as np
import argparse
from lib.dcc import IDEC
from lib.datasets import MNIST, FashionMNIST, Reuters
from sklearn.cluster import KMeans
from lib.utils import detect_wrong


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Instance Difficulty Constrained Clustering Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--pretrain', type=str, default="../model/mnist_sdae_weights.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    parser.add_argument('--use_pretrain', type=bool, default=True)
    args = parser.parse_args()

    # Load data
    mnist_train = MNIST('./dataset/mnist', train=True, download=True)
    mnist_test = MNIST('./dataset/mnist', train=False)
    X = mnist_train.train_data
    y = mnist_train.train_labels
    test_X = mnist_test.test_data
    test_y = mnist_test.test_labels
    
    # Set parameters
    ml_penalty, cl_penalty = 0.1, 1
    
    idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
        fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        test_X = fashionmnist_test.test_data
        test_y = fashionmnist_test.test_labels
        args.pretrain="../model/fashion_sdae_weights.pt"
        ml_penalty = 1
    elif args.data == "Reuters":
        reuters_train = Reuters('./dataset/reuters', train=True, download=False)
        reuters_test = Reuters('./dataset/reuters', train=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        test_X = reuters_test.test_data
        test_y = reuters_test.test_labels
        args.pretrain="../model/reuters10k_sdae_weights.pt"
        idec = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    if args.use_pretrain:
        idec.load_model(args.pretrain)
        
    # Print netowrk structure
    print(idec)

    # Construct Constraints
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])
    anchor, positive, negative = np.array([]), np.array([]), np.array([])
    
    # Provide instance guidance based on k-means results. High confidence (1) for correct instances.
    # Low confidence (0.1) for incorrect instances since k-means + AE does not achieve good results.
    latent = idec.encodeBatch(X).cpu().numpy()
    kmeans = KMeans(10, n_init=20)
    y_pred = kmeans.fit_predict(latent)
    instance_guidance = detect_wrong(y.cpu().numpy(), y_pred)
    instance_guidance = torch.tensor(instance_guidance, dtype=torch.float32).cuda()
    use_global = False
    
    # Train the network
    train_acc, train_nmi, epo = idec.fit(anchor, positive, negative, ml_ind1, ml_ind2, cl_ind1, cl_ind2, instance_guidance, use_global,  ml_penalty, cl_penalty, X, y,
                             lr=args.lr, batch_size=args.batch_size, num_epochs=args.epochs,
                             update_interval=args.update_interval, tol=1*1e-3)
    
    # Make prediction
    test_acc, test_nmi = idec.predict(test_X, test_y)

    # Report results
    print("Training Accuracy:", train_acc)
    print("Training NMI;", train_nmi)
    print("Training Epochs:", epo)
    print("Test Accuracy:", test_acc)
    print("Test NMI:", test_nmi)

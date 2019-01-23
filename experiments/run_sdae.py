import sys

sys.path.append("..")
import torch.utils.data
import argparse
from lib.stackedDAE import StackedDAE
from lib.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--pretrainepochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    args = parser.parse_args()

    # Load data for pre-training
    train_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
                      encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                      dropout=0)
    
    # Print the pre-train model structure
    print(sdae)
    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
                  num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="mse")
    
    # Train the stacked denoising autoencoder
    sdae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs, corrupt=0.2, loss_type="mse")
    
    # Save the weights as pre-trained model for IDEC/DEC/DCC
    sdae.save_model("model/sdae_mnist_weights.pt")

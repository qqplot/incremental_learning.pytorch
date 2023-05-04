import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, RandomSampler
import torchvision
from torchvision import datasets, transforms
import os
from inclearn.lib import data, factory, losses, network, utils
import clip
import numpy as np
import sys
import tqdm



def get_zero_acc(model_name):
    model, preprocess = clip.load(model_name, device=device)
    out_dim = model.text_projection.shape[1]

    print("Features dimension is {}.".format(out_dim))

    model.to(device)
    model.eval()
    torch.cuda.empty_cache()

    class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

    cifar100_train = torchvision.datasets.CIFAR100(root="./data", train=True, transform=preprocess)
    cifar100_test = torchvision.datasets.CIFAR100(root="./data", train=False, transform=preprocess)

    # train_list = {}
    # test_list = {}
    # for i in range(100):
    #     train_list[i] = []
    #     test_list[i] = []
    # print("Building Train List...")
    # for idx, (X,y) in enumerate(cifar100_train):
    #     train_list[y].append(idx)    
    # print("Building Test List...")
    # for idx, (X,y) in enumerate(cifar100_test):
    #     test_list[y].append(idx)

    train_list = torch.load('./exps/zero_train/train_list.pth')
    test_list = torch.load('./exps/zero_train/test_list.pth')
    
    print("Obtaining feature means...")
    feature_means = np.zeros((100,out_dim))
    for class_idx in class_order:
        class_loader = DataLoader(cifar100_train, shuffle=False, batch_size=500, 
                                sampler=SubsetRandomSampler(train_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.to(device)

                features = model.encode_image(X).cpu().numpy()
                feature_means[class_idx] = features.mean(axis=0)

            torch.cuda.empty_cache()
    
    print("Obtaining test accuracy...\n")
    class_acc = []

    for class_idx in class_order:
        class_loader = DataLoader(cifar100_test, shuffle=False, batch_size=100, 
                                sampler=SubsetRandomSampler(test_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.numpy()

                features = model.encode_image(X).cpu().numpy()
                
                dists = -2 * (features@feature_means.T) + np.power(features, 2).sum(axis=1, keepdims=True) + np.power(feature_means, 2).sum(axis=1, keepdims=True).T
                preds = dists.argsort()[:,0]
                class_acc.append(np.sum(preds == y) / 100)

    print(class_acc)
    print()

    for inc in [10,5,2,1]:
        print("Incremental Class Number :", inc)
        avg_acc, last_acc = get_average_accuracy(class_acc,inc)
        print("    Average Accuracy :", avg_acc)
        print("    Last Accuracy    :", last_acc)


def get_average_accuracy(acc,inc):
    accs = []
    for i in range(50//inc+1):
        inc_acc = np.mean(acc[:50+i*inc])
        accs.append(inc_acc)

    return np.mean(accs), accs[-1]



if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model_name = sys.argv[1]

    print(f"Using {model_name} model\n")
    get_zero_acc(model_name)
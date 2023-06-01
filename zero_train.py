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


from inclearn.convnet import resnet



def get_zero_acc(model_name, n_samples):
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        out_dim = model.out_dim
    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=True)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        out_dim = model.out_dim
    elif model_name == 'resnet18_fc':
        model = torchvision.models.resnet18(pretrained=True)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        out_dim = model.fc.out_features
    elif model_name == 'resnet50_fc':
        model = torchvision.models.resnet50(pretrained=True)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        out_dim = model.fc.out_features
    else:
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
        class_loader = DataLoader(cifar100_train, shuffle=False, batch_size=n_samples, 
                                sampler=SubsetRandomSampler(train_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.to(device)

                if model_name in ['resnet18', 'resnet50']:
                    features = model(X)["features"].cpu().numpy()
                elif model_name in ['resnet18_fc', 'resnet50_fc']:
                    features = model(X).cpu().numpy()
                else:
                    features = model.encode_image(X).cpu().numpy()
                feature_means[class_idx] = features.mean(axis=0)

            torch.cuda.empty_cache()
    
    print("Obtaining test accuracy...\n")
    class_accs = []
    
    MAX = 100000000000000

    class_nums = np.arange(50,101,1)

    for i, class_idx in enumerate(class_order):
        class_loader = DataLoader(cifar100_test, shuffle=False, batch_size=100, 
                                sampler=SubsetRandomSampler(test_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.numpy()

                if model_name in ['resnet18', 'resnet50']:
                    features = model(X)["features"].cpu().numpy()
                elif model_name in ['resnet18_fc', 'resnet50_fc']:
                    features = model(X).cpu().numpy()
                else:
                    features = model.encode_image(X).cpu().numpy()

                dists = -2 * (features@feature_means.T) + np.power(features, 2).sum(axis=1, keepdims=True) + np.power(feature_means, 2).sum(axis=1, keepdims=True).T

                class_acc = []
                for class_num in class_nums:
                    if class_num < i:
                        class_acc.append(0.)
                        continue
                    dists_copy = dists.copy()
                    dists_copy[:,class_order[class_num:]] = MAX
                    preds = dists_copy.argsort()[:,0]
                    class_acc.append(np.sum(preds == y) / 100)

                class_accs.append(class_acc)
    
    class_accs_np = np.array(class_accs)

    final = [model_name]

    for inc in [10,5,2,1]:
        inc_acc = []
        for i in np.arange(50,101,inc):
            inc_acc.append(np.mean(class_accs_np[:i,i-50]))

        print("Incremental Class Number :", inc)
        print("    Average Accuracy :", np.mean(inc_acc))
        print("    Last Accuracy    :", inc_acc[-1])
        
        if inc == 10: final.append(inc_acc[-1] * 100)
        final.append(np.mean(inc_acc) * 100)

    print()
    for item in final:
        print(item, end=",")


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
    n_samples = int(sys.argv[2]) # 500 for full_samples, 20 for 20_samples

    print(f"Using {model_name} model")
    print(f"Using {n_samples} samples\n")
    get_zero_acc(model_name, n_samples)
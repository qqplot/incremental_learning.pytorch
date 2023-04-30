import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, RandomSampler
import torchvision
from torchvision import datasets, transforms
import os
from inclearn.lib import data, factory, losses, network, utils
import clip
import numpy as np


def closest_to_mean(features, nb_examplars):
    features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]

def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    classifier_kwargs = {
    'type': 'cosine',
    'scaling': 3.0,
    'proxy_per_class': 10,
    'distance': 'neg_stable_cosine_distance'
    }
  
    model, preprocess = clip.load("ViT-B/32", device=device)
    out_dim = model.text_projection.shape[1]
        
    print("Features dimension is {}.".format(out_dim))

    model.to(device)
    model.eval()
    torch.cuda.empty_cache()

    class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]


    cifar100 = torchvision.datasets.CIFAR100(root="./data/cifar100", transform=preprocess)
    

    class_list = {}
    for i in range(100):
        class_list[i] = []
    for idx, (X,y) in enumerate(cifar100):
        class_list[y].append(idx)



    feature_means = {}

    for class_idx in class_order:
        
        if str(class_idx) not in feature_means.keys():
            feature_means[str(class_idx)] = []

        class_loader = DataLoader(cifar100, shuffle=False, batch_size=500, 
                                sampler=SubsetRandomSampler(class_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.to(device)

                features = model.encode_image(X)
                feature_means[str(class_idx)].append(features.mean(axis=0))

            torch.cuda.empty_cache()


    acc_list = []
    for class_idx in class_order:
        class_loader = DataLoader(cifar100, shuffle=False, batch_size=500, 
                                sampler=SubsetRandomSampler(class_list[class_idx]))
        with torch.no_grad():
            for X, y in class_loader:
                X, y = X.to(device), y.to(device)

                features = model.encode_image(X)
                
                preds = _l2_distance(features, feature_means[class_idx]).argsort()[:1]
                print(preds)


            torch.cuda.empty_cache()

            

    
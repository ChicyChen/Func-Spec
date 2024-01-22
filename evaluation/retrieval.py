import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import copy
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer, required
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import top_k_accuracy_score


"""# KNN evaluation to track classification accuray through ssl pretraining"""


class Retrieval():
    def __init__(self, model, k, device, num_seq=10, byol=True):
        super(Retrieval, self).__init__()
        self.k = k
        self.device = device
        self.num_seq = num_seq
        self.model = model
        self.model.eval()
        self.byol = byol

    def knn(self, features, labels, k=1):
        """
        Evaluating knn accuracy in feature space.
        Calculates only top-1 accuracy (returns 0 for top-5)
        Args:
            features: [... , dataset_size, feat_dim]
            labels: [... , dataset_size]
            k: nearest neighbours
        Returns: train accuracy, or train and test acc
        """
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features_np = features.cpu().view(-1, feature_dim).numpy()
            labels_np = labels.cpu().view(-1).numpy()
            # fit
            # print(features_np.shape, labels_np.shape)
            self.cls = KNeighborsClassifier(k, metric="cosine").fit(features_np, labels_np)
            # self.cls2 = NearestNeighbors(k, metric="cosine").fit(features_np, labels_np)
            acc = self.cls.score(features_np, labels_np)
            
        return acc
    
    def eval(self, features, labels, y):
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features = features.cpu().view(-1, feature_dim).numpy()
            labels = labels.cpu().view(-1).numpy()
            # acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
            # acc = self.cls.score(features, labels)


            pred_nei = self.cls.kneighbors(features, n_neighbors=1, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc1 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=5, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc5 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=10, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc10 = acc_num/total_num


            """
            scores = self.cls.predict_proba(features) # B, NClass
            # print(labels.shape, scores.shape)
            acc1 = top_k_accuracy_score(labels, scores, k=1)
            acc5 = top_k_accuracy_score(labels, scores, k=5)
            acc10 = top_k_accuracy_score(labels, scores, k=10)
            acc50 = top_k_accuracy_score(labels, scores, k=50)
            """
        return acc1, acc5, acc10
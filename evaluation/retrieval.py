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
        print("The dimension of feature is:", feature_dim)
        with torch.no_grad():
            features = features.cpu().view(-1, feature_dim).numpy()
            print("in the eval function, the h_total after reshape is: ", features.shape)
            labels = labels.cpu().view(-1).numpy()
            # acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
            # acc = self.cls.score(features, labels)


            pred_nei = self.cls.kneighbors(features, n_neighbors=1, return_distance=False)
            # print("===============================shape of predicted neighbours is: ",pred_nei.shape)
            # print("===============================shape of labels: ", labels.shape)
            # print("============================just want to verfiy what is labels[i]: ", labels[0])
            total_num = pred_nei.shape[0]
            # print("=========================total number is: ", total_num)
            # print("===========================now pred_nei is using n_neighbors = 1,nei_indices are:", pred_nei[0])
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc1 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=5, return_distance=False)
            # print("===========================now pred_nei is using n_neighbors = 1,nei_indices are:", pred_nei[0])
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        match = True
                        break
            acc5 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=10, return_distance=False)
            # print("===========================now pred_nei is using n_neighbors = 10,nei_indices are:", pred_nei[0])
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                # match = False
                # predicted_labels_10nei = torch.zeros((10,1))
                # index = 0
                for j in nei_indices:
                    # predicted_labels_10nei[index] = y[j]
                    # index = index + 1
                    if y[j] == labels[i]:
                        acc_num += 1
                        # match = True
                        break
            #     if not match:
            #         print("not matched knn k = 10, the first 5 predicted labels:", predicted_labels_10nei[0:5])
            #         print("not matched knn k = 10, the second 5 predicted labels:", predicted_labels_10nei[5:10])
            #         print("not matched knn k = 10",torch.unique(predicted_labels_10nei))
            #         print("then in this unmatched case what is the value of labels[i]????????", labels[i])
            # print("total_num of using n_neighbours = 10 is:", total_num)
            # print("acc_num of using n_neighbour = 10 is:", acc_num)
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
    

class Retrieval2Encoders():
    def __init__(self, model1, model2, k, device, num_seq=10, byol=True):
        super(Retrieval2Encoders, self).__init__()
        self.k = k
        self.device = device
        self.num_seq = num_seq
        self.model1 = model1
        self.model2 = model2
        self.model1.eval()
        self.model2.eval()
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
            # print("===============================shape of predicted neighbours is: ",pred_nei.shape)
            # print("===============================shape of labels: ", labels.shape)
            # print("============================just want to verfiy what is labels[i]: ", labels[0])
            total_num = pred_nei.shape[0]
            # print("=========================total number is: ", total_num)
            # print("===========================now pred_nei is using n_neighbors = 1,nei_indices are:", pred_nei[0])
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc1 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=5, return_distance=False)
            # print("===========================now pred_nei is using n_neighbors = 1,nei_indices are:", pred_nei[0])
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        match = True
                        break
            acc5 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=10, return_distance=False)
            # print("===========================now pred_nei is using n_neighbors = 10,nei_indices are:", pred_nei[0])
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                # match = False
                # predicted_labels_10nei = torch.zeros((10,1))
                # index = 0
                for j in nei_indices:
                    # predicted_labels_10nei[index] = y[j]
                    # index = index + 1
                    if y[j] == labels[i]:
                        acc_num += 1
                        # match = True
                        break
            #     if not match:
            #         print("not matched knn k = 10, the first 5 predicted labels:", predicted_labels_10nei[0:5])
            #         print("not matched knn k = 10, the second 5 predicted labels:", predicted_labels_10nei[5:10])
            #         print("not matched knn k = 10",torch.unique(predicted_labels_10nei))
            #         print("then in this unmatched case what is the value of labels[i]????????", labels[i])
            # print("total_num of using n_neighbours = 10 is:", total_num)
            # print("acc_num of using n_neighbour = 10 is:", acc_num)
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
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    
'''
import os
import sys
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.linalg import norm
from PIL import Image


def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def detect_wrong(y_true, y_pred):
    """
    Simulating instance difficulty constraints. Require scikit-learn installed
    
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        A mask vector M =  1xn which indicates the difficulty degree
        We treat k-means as weak learner and set low confidence (0.1) for incorrect instances.
        Set high confidence (1) for correct instances.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    mapping_dict = {}
    for pair in ind:
        mapping_dict[pair[0]] = pair[1]
    wrong_preds = []
    for i in range(y_pred.size):
        if mapping_dict[y_pred[i]] != y_true[i]:
            wrong_preds.append(-0.1)   # low confidence -0.1 set for k-means weak learner
        else:
            wrong_preds.append(1)
    return np.array(wrong_preds)


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links. 
    
    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = y.to(torch.device("cpu"))
    y = y.numpy()
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def generate_mnist_triplets(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    mnist_embedding = np.load("../model/mnist_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(mnist_embedding[tmp_anchor_index]-mnist_embedding[tmp_pos_index], 2)
        neg_distance = norm(mnist_embedding[tmp_anchor_index]-mnist_embedding[tmp_neg_index], 2)
        # 35 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 35:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_triplet_constraints_continuous(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    fashion_embedding = np.load("../model/fashion_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(fashion_embedding[tmp_anchor_index]-fashion_embedding[tmp_pos_index], 2)
        neg_distance = norm(fashion_embedding[tmp_anchor_index]-fashion_embedding[tmp_neg_index], 2)
        # 80 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 80:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)

import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import random
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import csv
import pandas as pd
from more_itertools import powerset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sys

class KNN:
    def __init__(self, k = 3, encoder = "resnets", distance_metric = "eucledian", weighing_scheme = "equal_for_all", feature_scalling = "", splitting_ratio = 0.2):
        self.k = k
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.weighing_sceme = weighing_scheme
        self.feature_scalling = feature_scalling
        self.splitting_ratio = splitting_ratio

    def train(self, data):
        self.data = data

    def predict(self, data_point):
        heap = []
        heapq.heapify(heap)
        for i in range(0, self.data.shape[0]):
            if self.distance_metric == "eucledian":
                if self.encoder == "resnets":
                    difference = self.data[i][1][0] - data_point
                elif self.encoder == "vit":
                    difference = self.data[i][2][0] - data_point
                squared = np.square(difference)
                sum = np.sum(squared)
                distance = math.sqrt(sum)
                heapq.heappush(heap, (distance, self.data[i][3]))
            if self.distance_metric == "manhattan":
                if self.encoder == "resnets":
                    difference = self.data[i][1][0] - data_point
                elif self.encoder == "vit":
                    difference = self.data[i][2][0] - data_point
                abs = np.abs(difference)
                sum = np.sum(abs)
                distance = sum
                heapq.heappush(heap, (distance, self.data[i][3]))
            if self.distance_metric == "cosine":
                if self.encoder == "resnets":
                    sq1 = np.square(self.data[i][1][0])
                if self.encoder == "vit":
                    sq1 = np.square(self.data[i][2][0])
                sum1 = np.sum(sq1)
                sum1 = math.sqrt(sum1)
                sq2 = np.square(data_point)
                sum2 = np.sum(sq2)
                sum2 = math.sqrt(sum2)
                if self.encoder == "resnets":
                    mul = self.data[i][1][0] * data_point
                if self.encoder == "vit":
                    mul = self.data[i][2][0] * data_point
                sum = np.sum(mul)
                distance = 1 - sum / (sum1*sum2)
                heapq.heappush(heap, (distance, self.data[i][3]))
            if self.distance_metric == "l1":
                if self.encoder == "resnets":
                    distance = np.sum(np.abs(self.data[i][1][0] - data_point))
                if self.encoder == "vit":
                    distance = np.sum(np.abs(self.data[i][2][0] - data_point))
                heapq.heappush(heap, (distance, self.data[i][3]))
            if self.distance_metric == "l2":
                if self.encoder == "resnets":
                    distance = np.linalg.norm(self.data[i][1][0] - data_point)
                if self.encoder == "vit":
                    distance = np.linalg.norm(self.data[i][2][0] - data_point)
                heapq.heappush(heap, (distance, self.data[i][3]))
        dict = {}
        for i in range(0,self.k):
            top = heapq.heappop(heap)
            value = dict.get(top[1])
            if value is not None:
                dict[top[1]] += 1
            else:
                dict[top[1]] = 1
        max = 0
        label = ""
        for key in dict:
            value = dict[key]
            if value > max:
                max = value
                label = key
        return label

    def model_metrics(self, data):
        validation_set = data
        predicted = []
        true = []
        for i in range(0,validation_set.shape[0]):
            if self.encoder == "resnets":
                predicted.append(self.predict(validation_set[i][1][0]))
            if self.encoder == "vit":
                predicted.append(self.predict(validation_set[i][2][0]))
            true.append(validation_set[i][3])
        accuracy = metrics.accuracy_score(true, predicted)
        average_recall = metrics.recall_score(true, predicted, average='macro', zero_division=1)
        recall_per_class = metrics.recall_score(true, predicted, average=None, zero_division=1)
        average_f1 = metrics.f1_score(true, predicted, average='macro', zero_division=1)
        f1_per_class = metrics.f1_score(true, predicted, average=None, zero_division=1)
        average_precision = metrics.precision_score(true, predicted, average='macro', zero_division=1)
        precision_per_class = metrics.precision_score(true, predicted, average=None, zero_division=1)
        print("accuracy:", accuracy)
        print("average_recall:", average_recall)
        # print("recall_per_class:", recall_per_class)
        print("average_f1:", average_f1)
        # print("f1_per_class:", f1_per_class)
        print("average_precision:", average_precision)
        # print("precision_per_class:", precision_per_class)
        # print("true:", true)
        # print("predicted:", predicted)
        return accuracy

if len(sys.argv) > 1:
    user_input = sys.argv[1]
    print(user_input)
else:
    print("No input provided.")

data = np.load('data.npy', allow_pickle=True)
test_data = np.load(user_input, allow_pickle=True)

data_gameid = {}
for i in range(0, data.shape[0]):
    if data[i][0] in data_gameid:
        print("Invalid Data!")
    else:
        data_gameid[data[i][0]] = 1

labels = data[:, 3]
label_to_index = {}
index_to_label = {}
ct = 0
for label in labels:
    if label not in label_to_index:
        label_to_index[label] = ct
        index_to_label[ct] = label
        ct += 1
for i in range(0, data.shape[0]):
    data[i][3] = label_to_index[data[i][3]]
for i in range(0, test_data.shape[0]):
    test_data[i][3] = label_to_index[test_data[i][3]]

model = KNN(splitting_ratio=0.2, k=7, encoder="vit", distance_metric="l1")
model.train(data)
model.model_metrics(test_data)
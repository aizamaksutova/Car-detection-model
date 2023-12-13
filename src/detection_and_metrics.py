import glob
import os
import torch
import numpy as np
import random
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations
from PIL import Image
from torchvision import models
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import copy

class CustomDataset(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
        self.X_train = X_train.detach().cpu().numpy()
        self.y_train = y_train.detach().cpu().numpy()

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]


def find_square_of_intersection(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
 
    if x2 >= x3 and x4 >= x1 and y2 >= y3 and y4 >= y1:
        x5 = max(x1, x3)
        y5 = max(y1, y3)
        x6 = min(x2, x4)
        y6 = min(y2, y4)
        return (x5 - x6) * (y5 - y6)
    return 0


# ============================== 1 Classifier model ============================


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    from torch.nn import Sequential
    return Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

def fit_cls_model(X_train, y_train):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    model = get_cls_model((40, 100, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    batch_size = 1
    num_epochs = 3
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        acc = 0
        pbar = tqdm(train_loader, desc=f'Training {epoch}/{num_epochs}')
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            acc += torch.sum(predictions.argmax(axis=1) == y_batch) / y_batch.shape[0]
            running_loss += loss.item() * X_batch.shape[0]
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss_Train: {running_loss / len(train_dataset)}, Accuracy: {acc / len(train_dataset)}")
    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """

    detection_model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(7, 22)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1),
        )

    detection_model.eval()
    cls_model.eval()


    for i in range(11):
        if i < 8:
            detection_model[i].load_state_dict(cls_model[i].state_dict())
        elif i == 8:
            state_dict = cls_model[i + 1].state_dict()
            state_dict['weight'] = state_dict['weight'].reshape(512, 64, 7, 22)
            detection_model[i].load_state_dict(state_dict)
        elif i == 10:
            state_dict = cls_model[i + 1].state_dict()
            state_dict['weight'] = state_dict['weight'].reshape(2, 512, 1, 1)
            detection_model[i].load_state_dict(state_dict)
              
    return detection_model

# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    predictions = {}
    detection_model.eval()
    n_rows, n_cols = [40, 100]
    for filename, image in dictionary_of_images.items():
        predictions[filename] = list()
        h, w = image.shape[0], image.shape[1]
        poolings_coef = 4
        image = np.pad(image, ((0, max(0, 220 - h)), (0, max(0, 370 - w))))[np.newaxis, np.newaxis, ...]
        prediction_map = detection_model(torch.from_numpy(image)).detach().numpy()[0, 1][:h // poolings_coef, :w //poolings_coef]
        for m in range(prediction_map.shape[0]):
            for n in range(prediction_map.shape[1]):
                confidence = prediction_map[m, n]
                predictions[filename].append([m * poolings_coef, n * poolings_coef, n_rows, n_cols, confidence])
    return predictions


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    rect1 = first_bbox[0], first_bbox[1], first_bbox[0] + first_bbox[2], first_bbox[1] + first_bbox[3]
    rect2 = second_bbox[0], second_bbox[1], second_bbox[0] + second_bbox[2], second_bbox[1] + second_bbox[3]
    square_of_intersection = find_square_of_intersection(rect1, rect2)
    """
    
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    return square_of_intersection / (first_bbox[2] * first_bbox[3] + second_bbox[2] * second_bbox[3] - square_of_intersection)

def square_of_rectangular_trapezoid(rect_1, rect_2):
    return (rect_2[1] + rect_1[1]) * (rect_2[0] - rect_1[0]) / 2

# =============================== 6 AUC ========================================



def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    tp = list()
    total = list()
    all_detections = 0
    for filename, detections in pred_bboxes.items():
        ground_truth = gt_bboxes[filename].copy()
        all_detections += len(ground_truth)
        for detection in sorted(detections, key=lambda x: -x[4]):
            best_score = 0
            for i, true_detection in enumerate(ground_truth):
                iou_score = calc_iou(detection[:4], true_detection)
                if iou_score > best_score:
                    best_score = iou_score
                    index_of_best_score = i
            if best_score >= 0.5:
                tp.append(detection[4])
                del ground_truth[index_of_best_score]
            total.append(detection[4])


    total.sort(reverse = True)
    tp.sort(reverse = True)

    pr_curve = [[0, 1]]
    j = 0
    i = 0
    while i < len(total):
        c = total[i]
        if i + 1 < len(total) and total[i] == total[i + 1]:
            i += 1
        else:
            while j < len(tp) and tp[j] >= c:
                j += 1
            precision = j / all_detections
            recall = j / (i + 1)
            pr_curve.append([precision, recall, c])
            i += 1

    roc_auc = 0
    for i in range(len(pr_curve) - 1):
        roc_auc += square_of_rectangular_trapezoid(pr_curve[i], pr_curve[i + 1])
    return roc_auc

# =============================== 7 NMS ========================================


def nms(detections_dictionary, iou_thr = 0.2):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    nms = dict()
    for filename, detections in detections_dictionary.items():
        sorted_detections = sorted(detections, key=lambda x: -x[4])
        j = 0
        nms[filename] = list()
        while j < len(sorted_detections):
            detection = sorted_detections[j]
            nms[filename].append(detection)
            i = j + 1
            while i < len(sorted_detections):
                if calc_iou(sorted_detections[i][:4], sorted_detections[j][:4]) >= iou_thr:
                    del sorted_detections[i]
                else:
                    i += 1
            j += 1
    return nms

# # if __name__ == '__main__':
# #     model torch.load('tensors.pt', map_location=torch.device('cpu'))
# #     print()
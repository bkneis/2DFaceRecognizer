import os
import random
import sys
from json import JSONDecodeError

import requests

import cv2
import numpy as np
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp


def __compute_roc(confidences, labels, classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], confidences[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), confidences.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def compute_roc(confidences, labels, classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], confidences[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), confidences.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def mock_openface(type, subject, i):
    base_path = 'http://localhost:5000/identify?image=./data/ps_test_v2/%s/%s-%s.png'
    res = requests.get(base_path % (subject, type, i))

    try:
        json = res.json()
    except JSONDecodeError:
        raise ValueError
    return np.array(json['data'][1]), json['data'][2]

    # if random.randint(1, 101) % 2:
    #     return np.array([0.04, 0.01, 0.09, 0.8]), ['1001', '1002', '1003', '1004']
    # else:
    #     return np.array([0.12, 0.32, 0.11, 0.84]), ['1001', '1002', '1003', '1004']


def classify(type, subject, i):
    # Send image to openface
    # if subject is not openface response return False
    # else return confidence
    preds, labels = mock_openface(type, subject, i)
    idx = labels.index(subject)
    binary_labels = np.zeros(len(labels))
    binary_labels[idx] = 1
    return preds, binary_labels, labels


def main(root_dir):

    subject_dirs = get_immediate_subdirectories(root_dir)

    intense_binary_labels = []
    intense_labels = None
    intense_scores = []

    left_binary_labels = []
    left_labels = None
    left_scores = []

    normal_binary_labels = []
    normal_labels = None
    normal_scores = []

    for subject in subject_dirs:
        subject_dir = os.path.join(root_dir, subject)
        i = 0
        while True:
            intense_img = cv2.imread(os.path.join(subject_dir, 'intense-%s.png' % i))
            left_img = cv2.imread(os.path.join(subject_dir, 'left-%s.png' % i))
            normal_img = cv2.imread(os.path.join(subject_dir, 'average-%s.png' % i))

            if intense_img is None or left_img is None or normal_img is None:
                break

            try:
                preds, labels, intense_labels = classify('intense', subject, i)
                intense_binary_labels.append(labels)
                intense_scores.append(preds)
            except ValueError:
                print('Subject %s was not detected' % subject)

            try:
                preds, labels, left_labels = classify('left', subject, i)
                left_binary_labels.append(labels)
                left_scores.append(preds)
            except ValueError:
                print('Subject %s was not detected' % subject)

            try:
                preds, labels, normal_labels = classify('average', subject, i)
                normal_binary_labels.append(labels)
                normal_scores.append(preds)
            except ValueError:
                print('Subject %s was not detected' % subject)

            i += 1

    # print(np.array(normal_scores),
    #       np.array(normal_binary_labels), np.array(normal_labels))
    #
    # print(np.array(normal_scores).shape,
    #       np.array(normal_binary_labels).shape, np.array(normal_labels).shape)

    np.savetxt('scores.txt', np.array(normal_scores), delimiter=',')
    # np.savetxt('classess.txt', np.array(normal_labels), delimiter=',')
    np.savetxt('labels.txt', np.array(normal_binary_labels), delimiter=',')

    intense_fpr, intense_tpr, intense_roc = compute_roc(np.array(intense_scores),
                              np.array(intense_binary_labels), np.array(intense_labels))

    left_fpr, left_tpr, left_roc = compute_roc(np.array(left_scores),
                           np.array(left_binary_labels), np.array(left_labels))

    normal_fpr, normal_tpr, normal_roc = compute_roc(np.array(normal_scores),
                             np.array(normal_binary_labels), np.array(normal_labels))

    # Plot all ROC curves
    plt.figure()
    lw = 2

    plt.plot(normal_fpr["micro"], normal_tpr["micro"],
             label='Averaged - ROC curve (area = {0:0.2f})'
                   ''.format(normal_roc["micro"]),
             color='darkorange')

    plt.plot(left_fpr["micro"], left_tpr["micro"],
             label='Directional - ROC curve (area = {0:0.2f})'
                   ''.format(left_roc["micro"]),
             color='darkred')

    plt.plot(intense_fpr["micro"], intense_tpr["micro"],
             label='Intense - ROC curve (area = {0:0.2f})'
                   ''.format(intense_roc["micro"]),
             color='darkgreen')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw, color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Photoface Database Face Recognition Performance')
    plt.legend(loc="lower right")
    plt.savefig('/home/arthur/openface_results_%s.png' % time.strftime('%c'))
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])

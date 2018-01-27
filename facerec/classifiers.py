import numpy as np
import cv2


class StatModel(object):
    """parent class - starting point to add abstraction"""
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    """wrapper for OpenCV SimpleVectorMachine algorithm"""
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setC(C)
        self.model.setGamma(gamma)
        # svm.setDegree(0.0)
        # svm.setGamma(0.0)
        # svm.setCoef0(0.0)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


class KNearest(StatModel):
    def __init__(self, k=1):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, self.k)
        print('knn', _retval, results, _neigh_resp, _dists)

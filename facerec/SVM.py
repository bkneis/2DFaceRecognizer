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
    def __init__(self):
        self.model = cv2.ml.SVM_create()
        # self.model.setType(cv2.ml.SVM_C_SVC)
        # self.model.setKernel(cv2.ml.SVM_LINEAR)
        # self.model.setC(1)
        # svm.setDegree(0.0)
        # svm.setGamma(0.0)
        # svm.setCoef0(0.0)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)

    def train(self, samples, responses):
        self.model.trainAuto(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)

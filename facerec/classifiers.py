import cv2


class Classifier(object):
    """Base Class to wrap OpenCV StatModel
    This class provides a base for all opencv classifiers and wraps certain functions to provide
    the model's API to the classifier
    @todo use base function to ensure predict is overwritten
    """
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)


class SVM(Classifier):
    def __init__(self, C=1.0, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setC(C)
        self.model.setGamma(gamma)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


class MLP(Classifier):
    """TODO set params"""
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def predict(self, samples):
        pass


class DecisionTrees(Classifier):
    """TODO set params"""
    def __init__(self):
        self.model = cv2.ml.DTrees_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


class KNearest(Classifier):
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, self.k)
        if self.k > 1:
            return _dists[0][0], int(_neigh_resp[0][0])
        return _dists[0], int(_neigh_resp[0])


class LogisticRegression(Classifier):
    """TODO set params for defaults and return correctr results"""
    def __init__(self, iter, learning_rate, batch_size, regularization):
        self.model = cv2.ml.LogisticRegression_create()
        self.model.setIterations(iter)
        self.model.setLearningRate(learning_rate)
        self.model.setMiniBatchSize(batch_size)
        self.model.setRegularization(regularization)

    def predict(self, samples):
        retval, results = cv2.model.predict(samples)
        print('results', results)
        return results


class NormalBayes(Classifier):

    def __init__(self):
        self.model = cv2.ml.NormalBayesClassifier_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, outputs, output_probs = self.model.predictProb(samples)
        return outputs[0][0]


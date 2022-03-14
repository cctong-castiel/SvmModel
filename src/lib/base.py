from abc import abstractmethod
import numpy as np

class _Base():

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:

        # normalizing all the n features of X
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X

    @abstractmethod
    def sgd(self):
        pass

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z: np.ndarray):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def gradients(self, 
                  X: np.ndarray,
                  y: np.array):
        
        dW = np.sum(self.C * y * X) / len(X)

        db = np.sum(self.C * y) / len(y)

        return dW, db

    def loss(self,
             X: np.ndarray,
             y: np.array):
        
        half_w_d = 0.5 * np.dot(self.W, self.W)
        total_d = self.C * np.sum( y * np.dot(self.W, X) - 1)
        loss = half_w_d - total_d

        return loss
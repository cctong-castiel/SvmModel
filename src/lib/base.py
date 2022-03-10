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

    def gradients(self, 
                  X: np.ndarray,
                  y: np.array):
        
        dW = np.sum(self.lagrange * y * X)

        db = np.sum(self.lagrange * y)

        return dW, db

    def loss(self,
             X: np.ndarray,
             y: np.array):
        
        half_w_d = 0.5 * np.dot(self.W, self.W)
        total_d = np.sum(self.lagrange * y * np.dot(self.W, X) - 1)
        loss = half_w_d + total_d

        return loss
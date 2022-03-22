from abc import abstractmethod
from typing import Union
import numpy as np

ZERO = 1e-7

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
    def softmax(z: np.ndarray) -> np.ndarray:
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    # def kernel_rbf(self, x1: np.array, x2: np.array) -> float:

    #     trnorms1 = np.mat([(v * v.T)[0, 0] for v in x1]).T
    #     trnorms2 = np.mat([(v * v.T)[0, 0] for v in x2]).T

    #     k1 = trnorms1 * np.mat(np.ones((x2.shape[0], 1), dtype=np.float64)).T

    #     k2 = np.mat(np.ones((x1.shape[0], 1), dtype=np.float64)) * trnorms2.T

    #     k = k1 + k2

    #     k -= 2 * np.mat(x1 * x2.T)

    #     k *= - 1./(2 * np.power(self.gamma, 2))

    #     return np.exp(k)

    def kernel_rbf(self, x1: np.ndarray, x2: np.ndarray) -> float:
        X_norm = np.sum(x1 ** 2, axis=-1)
        return -0.5 * np.exp(-self.gamma * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(x1, x2)))

    @staticmethod
    def kernel_linear(x1: np.array, x2: np.array) -> float:
        return np.dot(x1, x2)

    def kernel_poly(self, x1: np.array, x2: np.array) -> float:
        return np.dot(x1, x2) ** self.degree

    def distances(self, 
                  X: np.ndarray,
                  y: np.array,
                  with_lagrange: bool=True):

        print(f"shape of self.W: {self.W.shape}")
        
        distances = y * np.dot(X, self.W) - 1

        print(f"shape of distances: {distances.shape}")
        

        if with_lagrange:
            # if distance is more than 0, sample is not on the support vector
            # lagrange multiplier will be 0
            distances[distances > 0] = 0
        
        return distances

    def gradients(self, 
                  X: np.ndarray,
                  y: np.array,
                  alpha: np.array) -> np.ndarray:
        
        distances = self.distances(X, y)

        dW = np.zeros(len(self.W))

        for ind, d in enumerate(distances):
            if d == 0:
                di = self.W
            else:
                di = self.W - (self.C * y[ind] * X[ind])

            dW += di

        print(f"y shape in gradients: {y.shape}")
        db = np.sum(self.C * alpha * y)

        return dW / len(X), db

    def linear_loss(self,
             X: np.ndarray,
             y: np.array,
             alpha: Union[np.array, None]) -> float:
        
        half_w_d = 0.5 * np.dot(self.W, self.W)
        total_d = self.C * np.sum( y * np.dot(X, self.W) - 1)
        loss = half_w_d - total_d

        return loss

    # def kernel_loss(self,
    #                 alpha: np.array,
    #                 X: np.ndarray,
    #                 t: np.array) -> float:

    #     loss = 0
    #     ind_sv = np.where(alpha > ZERO)[0]
    #     for i in ind_sv:
    #         for k in ind_sv:
    #             loss += np.sum(alpha) - 0.5 * alpha[i] * alpha[k] * t[i] * t[k] * self.kernels(X[i,:], X[k,:])

    #     # alpha[i] * alpha[k] = alpha * alpha.T

    #     return loss

    def kernel_loss(self,
             X: np.ndarray,
             y: np.array,
             alpha: Union[np.array, None]) -> float:

        print(f"shape of y: {y.shape}")
        print(f"shape of alpha: {alpha.shape}")
        print(f"shape of X: {X.shape}")

        half_w_d = 0.5 * np.dot(self.W, self.W)
        total_d = self.C * np.sum( alpha * y * self.kernels(X, X.T) - 1)
        loss = half_w_d - total_d

        return loss
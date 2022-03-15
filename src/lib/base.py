from abc import abstractmethod
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

    def kernel_rbf(self, x1: np.array, x2: np.array) -> float:

        trnorms1 = np.mat([(v * v.T)[0, 0] for v in x1]).T
        trnorms2 = np.mat([(v * v.T)[0, 0] for v in x2]).T

        k1 = trnorms1 * np.mat(np.ones((x2.shape[0], 1), dtype=np.float64)).T

        k2 = np.mat(np.ones((x1.shape[0], 1), dtype=np.float64)) * trnorms2.T

        k = k1 + k2

        k -= 2 * np.mat(x1 * x2.T)

        k *= - 1./(2 * np.power(self.gamma, 2))

        return np.exp(k)

    @staticmethod
    def kernel_linear(x1: np.array, x2: np.array) -> float:
        return np.dot(x1, x2)

    def kernel_poly(self, x1: np.array, x2: np.array) -> float:
        return np.dot(x1, x2) ** self.degree

    def gradients(self, 
                  X: np.ndarray,
                  y: np.array) -> np.ndarray:
        
        dW = np.sum(self.C * y * X) / len(X)

        db = np.sum(self.C * y) / len(y)

        return dW, db

    def linear_loss(self,
             X: np.ndarray,
             y: np.array) -> float:
        
        half_w_d = 0.5 * np.dot(self.W, self.W)
        total_d = self.C * np.sum( y * np.dot(self.W, X) - 1)
        loss = half_w_d - total_d

        return loss

    def kernel_loss(self,
                    alpha: np.array,
                    X: np.ndarray,
                    t: np.array) -> float:

        loss = 0
        ind_sv = np.where(alpha > ZERO)[0]
        for i in ind_sv:
            for k in ind_sv:
                loss += np.sum(alpha) - 0.5 * alpha[i] * alpha[k] * t[i] * t[k] * self.kernels(X[i,:], X[k,:])

        # alpha[i] * alpha[k] = alpha * alpha.T

        return loss
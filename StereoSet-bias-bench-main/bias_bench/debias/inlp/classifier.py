import numpy as np


class Classifier(object):
    """An abstract class for linear classifiers."""

    def __init__(self):
        pass

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_dev: np.ndarray,
        Y_dev: np.ndarray,
    ) -> float:
        """
        Returns:
            Accuracy on the dev set.
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """Returns:
        Final weights of the model, as np.ndarray.
        """
        raise NotImplementedError


class SKlearnClassifier(Classifier):
    def __init__(self, m):

        self.model = m

    def train_network(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_dev: np.ndarray,
        Y_dev: np.ndarray,
    ) -> float:
        """Returns:
        Accuracy score on the dev set / Pearson's R in the case of regression.
        """
        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """Returns:
        Final weights of the model, as np.ndarray.
        """
        w = self.model.coef_
        if len(w.shape) == 1:
            w = np.expand_dims(w, 0)

        return w

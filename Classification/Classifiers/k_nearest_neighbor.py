import time

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

from utilities import grid_search, report_scores, CrossValidation, evaluation
from data_setup import train_label, train_set, test_label, test_set


class KNearestNeighbors:
    """KNN"""

    def __init__(self, seed):
        self.knn = None
        self.knn_best_params = None
        self.knn_model_selection_time = None
        self.acc_knn_test = None
        self.acc_knn_train = None
        self.seed = seed
        self.param_dist = {"n_neighbors": [i for i in range(4, 32)]}

    def model_selection(self):
        """Model Selection"""
        knn = KNeighborsClassifier()

        start_time = time.time()
        self.knn_best_params = grid_search(model=knn, dict_params=self.param_dist, plot=True, train_set=train_set,
                                           train_label=train_label)
        end_time = time.time()
        self.knn_model_selection_time = end_time - start_time
        print("Model selection time", self.knn_model_selection_time, "sec")

        n_neighbors = self.knn_best_params["n_neighbors"]

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_set, train_label)

        """### Evaluation """

        train_pred_knn = knn.predict(train_set)
        test_pred_knn = knn.predict(test_set)

        self.acc_knn_train, self.acc_knn_test = evaluation(train_label=train_label,
                                                           train_pred=train_pred_knn,
                                                           test_label=test_label,
                                                           test_pred=test_pred_knn)

        report_scores(test_label, test_pred_knn)

        CrossValidation(knn, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_knn)
        disp.ax_.set_title('KNN Confusion Matrix')
        plt.show()

    def saving_results(self, outputs):
        """Saving results"""
        key = "knn"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.knn_model_selection_time
        outputs[key]["params"] = self.knn_best_params
        outputs[key]["model"] = self.knn
        outputs[key]["train_acc"] = self.acc_knn_train
        outputs[key]["test_acc"] = self.acc_knn_test

        return outputs

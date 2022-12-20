import time

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data_setup import train_label, test_label, test_set, train_set
import numpy as np

from utilities import grid_search, evaluation, report_scores, CrossValidation
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    """## Naive Bayes"""

    def __init__(self, seed):
        self.gnb_best_params = None
        self.acc_gnb_test = None
        self.acc_gnb_train = None
        self.gnb = None
        self.gnb_model_selection_time = None
        self.seed = seed
        self.param_dist = {'var_smoothing': np.logspace(0, -9, num=250)}

    def model_selection(self):
        """### Model Selection"""
        gnb = GaussianNB()

        start_time = time.time()
        self.gnb_best_params = grid_search(model=gnb, dict_params=self.param_dist, plot=True,
                                           train_set=train_set, train_label=train_label)
        end_time = time.time()

        self.gnb_model_selection_time = end_time - start_time
        print("Model selection time", self.gnb_model_selection_time, "sec")

        best_var_smoothing = self.gnb_best_params["var_smoothing"]

        """### Evaluation """
        start_time = time.time()

        self.gnb = GaussianNB(var_smoothing=best_var_smoothing)
        self.gnb.fit(train_set, train_label)

        end_time = time.time()
        print("Model fitting time", end_time - start_time, "sec")

        train_pred_gnb = gnb.predict(train_set)
        test_pred_gnb = gnb.predict(test_set)

        prediction_dict = {'gaussian_nb': test_pred_gnb}

        self.acc_gnb_train, self.acc_gnb_test = evaluation(train_label=train_label,
                                                           train_pred=train_pred_gnb,
                                                           test_label=test_label,
                                                           test_pred=test_pred_gnb)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_gnb)
        disp.ax_.set_title('Naive Bayes Confusion Matrix')
        plt.show()

        CrossValidation(gnb, train_set, train_label)

        report_scores(train_label, train_pred_gnb)
        report_scores(test_label, test_pred_gnb)

    def saving_results(self, outputs):
        key = "nbc"
        outputs[key] = {'model': self.gnb,
                        'train_acc': self.acc_gnb_train,
                        'test_acc': self.acc_gnb_test,
                        'model_selection_time': self.gnb_model_selection_time,
                        'params': self.gnb_best_params}
        return outputs

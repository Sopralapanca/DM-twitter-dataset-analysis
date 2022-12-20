import time

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

from data_setup import train_set, test_set, test_label, train_label
from utilities import random_search, evaluation, report_scores, CrossValidation


class SupportVectorClassifier:
    """## Support Vector Classifier"""

    def __init__(self, seed):
        self.svc_model_selection_time = None
        self.svc_best_params = None
        self.svc = None
        self.acc_svc_train = None
        self.acc_svc_test = None
        self.param_dist = {
            "C": [1, 10, 100, 1000],
            "gamma": [0.001, 0.0001],
            "degree": [2, 3, 4, 5],
            "kernel": ["linear", "poly", "rbf", "sigmoid"]}
        self.seed = seed

    def model_selection(self):
        """### Model Selection"""
        svc = SVC(random_state=self.seed, cache_size=1000)

        start_time = time.time()
        self.svc_best_params = random_search(model=svc, dict_params=self.param_dist, plot=True, train_label=train_label,
                                             train_set=train_set)
        end_time = time.time()
        self.svc_model_selection_time = end_time - start_time
        print("Model selection time", self.svc_best_params, "sec")

        C = self.svc_best_params["C"]
        degree = self.svc_best_params["degree"]
        kernel = self.svc_best_params["kernel"]
        gamma = self.svc_best_params["gamma"]
        # class_weight_result   = svc_best_params["class_weight"]

        self.svc = SVC(C=C,
                       degree=degree,
                       kernel=kernel,
                       gamma=gamma,
                       # class_weight=class_weight_result
                       )

        self.svc.fit(train_set, train_label)

        train_pred_svc = self.svc.predict(train_set)
        test_pred_svc = self.svc.predict(test_set)

        """### Evaluation"""

        self.acc_svc_train, self.acc_svc_test = evaluation(train_label=train_label,
                                                           train_pred=train_pred_svc,
                                                           test_label=test_label,
                                                           test_pred=test_pred_svc)

        report_scores(train_label, train_pred_svc)

        report_scores(test_label, test_pred_svc)

        CrossValidation(svc, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_svc)
        disp.ax_.set_title('Support Vector Machine Confusion Matrix')
        plt.show()

    def saving_results(self, outputs):
        outputs["svc_model_selection_time"] = self.svc_model_selection_time
        outputs["svc_best_params"] = self.svc_best_params
        outputs["acc_svc_train"] = self.acc_svc_train
        outputs["acc_svc_test"] = self.acc_svc_test
        outputs["svc"] = self.svc
        return outputs

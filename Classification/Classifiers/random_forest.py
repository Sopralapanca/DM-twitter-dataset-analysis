from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from data_setup import train_label, train_set, test_label, test_set

from parameters import *
import time

from utilities import random_search, evaluation, report_scores, CrossValidation


class RandomForest:
    """Random Forest"""

    def __init__(self, seed):
        self.acc_rf_train = 0
        self.acc_rf_test = 0
        self.rf = None
        self.rf_best_params = None
        self.rf_model_selection_time = None
        self.param_dist = {"max_depth": max_depth,
                           "max_features": max_features_list,
                           "min_samples_split": min_sample_split_list,
                           "min_samples_leaf": min_sample_leaf_list,
                           "bootstrap": [True, False],
                           "criterion": criterion,
                           "class_weight": class_weight
                           }
        self.seed = seed

    def model_selection(self, n_estimators=30):
        """Model Selection"""
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=self.seed)
        start_time = time.time()
        self.rf_best_params = random_search(model=rf, dict_params=self.param_dist,
                                            train_label=train_label, train_set=train_set)
        end_time = time.time()
        self.rf_model_selection_time = end_time - start_time
        print("Model selection time", self.rf_model_selection_time, "sec")

        max_depth_result = self.rf_best_params["max_depth"]
        max_features_result = self.rf_best_params["max_features"]
        min_samples_split_result = self.rf_best_params["min_samples_split"]
        min_samples_leaf_result = self.rf_best_params["min_samples_leaf"]
        bootstrap_result = self.rf_best_params["bootstrap"]
        criterion_result = self.rf_best_params["criterion"]
        class_weight_result = self.rf_best_params["class_weight"]

        # fit a random forest classifiers using the best parameters found
        self.rf = RandomForestClassifier(n_estimators=n_estimators,
                                         criterion=criterion_result,
                                         max_features=max_features_result,
                                         max_depth=max_depth_result,
                                         min_samples_split=min_samples_split_result,
                                         min_samples_leaf=min_samples_leaf_result,
                                         bootstrap=bootstrap_result,
                                         class_weight=class_weight_result)

        # if I have a huge dataset consider all the data may be not so good, so we can set bootstrap = true
        self.rf.fit(train_set, train_label)

        train_pred_rf = self.rf.predict(train_set)
        test_pred_rf = self.rf.predict(test_set)

        self.acc_rf_train, self.acc_rf_test = evaluation(train_label=train_label,
                                                         train_pred=train_pred_rf,
                                                         test_label=test_label,
                                                         test_pred=test_pred_rf)

        report_scores(train_label, train_pred_rf)

        report_scores(test_label, test_pred_rf)

        CrossValidation(rf, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_rf)
        disp.ax_.set_title('Random Forest Confusion Matrix')
        plt.show()

    def saving_results(self, outputs):
        # Saving results
        key = "rf"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.rf_model_selection_time
        outputs[key]["params"] = self.rf_best_params
        outputs[key]["model"] = self.rf
        outputs[key]["train_acc"] = self.acc_rf_train
        outputs[key]["test_acc"] = self.acc_rf_test

        return outputs

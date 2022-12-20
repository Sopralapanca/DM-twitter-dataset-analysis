from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import time

from sklearn.metrics import ConfusionMatrixDisplay

from data_setup import train_set, test_set, test_label, train_label
from utilities import report_scores, CrossValidation, random_search, evaluation


class AdaBoost:
    """Ada Boosting Classifier"""

    def __init__(self, seed):
        self.dt_abc_model_selection_time = None
        self.dt_abc_best_params = None
        self.dt_abc = None
        self.acc_dt_abc_train = None
        self.acc_dt_abc_test = None
        self.rf_abc_model_selection_time = None
        self.rf_abc_best_params = None
        self.rf_abc = None
        self.acc_rf_abc_train = None
        self.acc_rf_abc_test = None
        self.param_dist = {
            "n_estimators": [10, 25, 50],
            "learning_rate": [0.5, 1.0, 1.5],
            "algorithm": ["SAMME", "SAMME.R"]}
        self.seed = seed

    def dt_based(self, dec_tree):
        """AdaBoost + DecisionTree"""
        dt_abc = AdaBoostClassifier(estimator=dec_tree, random_state=self.seed)

        start_time = time.time()
        self.dt_abc_best_params = random_search(model=dt_abc, dict_params=self.param_dist, train_label=train_label,
                                                train_set=train_set)
        end_time = time.time()
        self.dt_abc_model_selection_time = end_time - start_time
        print("Model selection time", self.dt_abc_model_selection_time, "sec")

        n_estimators_result = self.dt_abc_best_params["n_estimators"]
        learning_rate_result = self.dt_abc_best_params["learning_rate"]
        algorithm_result = self.dt_abc_best_params["algorithm"]

        dt_abc = AdaBoostClassifier(estimator=dec_tree, n_estimators=n_estimators_result,
                                         learning_rate=learning_rate_result, algorithm=algorithm_result,
                                         random_state=self.seed)
        self.dt_abc = dt_abc.fit(train_set, train_label)

        train_pred_dt_abc = dt_abc.predict(train_set)
        test_pred_dt_abc = dt_abc.predict(test_set)

        self.acc_dt_abc_train, self.acc_dt_abc_test = evaluation(train_label=train_label,
                                                                 train_pred=train_pred_dt_abc,
                                                                 test_label=test_label,
                                                                 test_pred=test_pred_dt_abc)

        report_scores(test_label, test_pred_dt_abc)

        CrossValidation(dt_abc, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_dt_abc)
        disp.ax_.set_title('AdaBoost Classifier + Decision Tree Confusion Matrix')
        plt.show()

    def rf_based(self, rf):
        """AdaBoost + Random Forest"""
        rf_abc = AdaBoostClassifier(estimator=rf, random_state=self.seed)

        start_time = time.time()
        self.rf_abc_best_params = random_search(model=rf_abc, dict_params=self.param_dist, train_label=train_label,
                                                train_set=train_set)
        end_time = time.time()
        self.rf_abc_model_selection_time = end_time - start_time
        print("Model selection time", self.rf_abc_model_selection_time, "sec")

        n_estimators_result = self.rf_abc_best_params["n_estimators"]
        learning_rate_result = self.rf_abc_best_params["learning_rate"]
        algorithm_result = self.rf_abc_best_params["algorithm"]

        rf_abc = AdaBoostClassifier(estimator=rf, n_estimators=n_estimators_result,
                                         learning_rate=learning_rate_result, algorithm=algorithm_result,
                                         random_state=self.seed)
        self.rf_abc = rf_abc.fit(train_set, train_label)

        train_pred_rf_abc = rf_abc.predict(train_set)
        test_pred_rf_abc = rf_abc.predict(test_set)

        self.acc_rf_abc_train, self.acc_rf_abc_test = evaluation(train_label=train_label,
                                                                 train_pred=train_pred_rf_abc,
                                                                 test_label=test_label,
                                                                 test_pred=test_pred_rf_abc)

        report_scores(train_label, train_pred_rf_abc)

        report_scores(test_label, test_pred_rf_abc)

        CrossValidation(rf_abc, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_rf_abc)
        disp.ax_.set_title('AdaBoost Classifier Confusion Matrix')
        plt.show()

    def dt_saving_results(self, outputs):
        """Saving results"""
        key = "dt_abc"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.dt_abc_model_selection_time
        outputs[key]["params"] = self.dt_abc_best_params
        outputs[key]["model"] = self.dt_abc
        outputs[key]["train_acc"] = self.acc_dt_abc_train
        outputs[key]["test_acc"] = self.acc_dt_abc_test

        return outputs

    def rf_saving_results(self, outputs):
        """Saving results"""
        key = "rf_abc"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.rf_abc_model_selection_time
        outputs[key]["params"] = self.rf_abc_best_params
        outputs[key]["model"] = self.rf_abc
        outputs[key]["train_acc"] = self.acc_rf_abc_train
        outputs[key]["test_acc"] = self.acc_rf_abc_test

        return outputs

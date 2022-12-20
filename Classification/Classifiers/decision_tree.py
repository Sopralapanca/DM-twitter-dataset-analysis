from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay
import time
import graphviz

from data_setup import train_label, train_set, test_set, test_label
from parameters import *
from utilities import random_search, CrossValidation, report_scores, evaluation


class DecisionTree:
    """Decision Tree"""

    def __init__(self, seed):
        self.dec_tree = None
        self.dt = None
        self.dt_best_params = None
        self.dt_model_selection_time = None
        self.acc_dt_test = None
        self.acc_dt_train = None
        self.seed = seed
        self.param_dist_dt = {"max_depth": max_depth,
                              "max_features": max_features_list,
                              "min_samples_split": min_sample_split_list,
                              "min_samples_leaf": min_sample_leaf_list,
                              "criterion": criterion,
                              "class_weight": class_weight}

    def model_selection(self):
        dec_tree = tree.DecisionTreeClassifier(random_state=self.seed)

        start_time = time.time()
        self.dt_best_params = random_search(model=dec_tree, dict_params=self.param_dist_dt,
                                            train_set=train_set, train_label=train_label)
        end_time = time.time()
        self.dt_model_selection_time = end_time - start_time
        print("Model selection time", self.dt_model_selection_time, "sec")

        max_depth_result = self.dt_best_params["max_depth"]
        max_features_result = self.dt_best_params["max_features"]
        min_samples_split_result = self.dt_best_params["min_samples_split"]
        min_samples_leaf_result = self.dt_best_params["min_samples_leaf"]
        criterion_result = self.dt_best_params["criterion"]
        class_weight_result = self.dt_best_params["class_weight"]

        """### Evaluation"""
        dec_tree = tree.DecisionTreeClassifier(criterion=criterion_result, max_depth=max_depth_result,
                                               min_samples_split=min_samples_split_result,
                                               max_features=max_features_result,
                                               min_samples_leaf=min_samples_leaf_result,
                                               class_weight=class_weight_result)

        self.dec_tree = dec_tree.fit(train_set, train_label)

        dot_data = tree.export_graphviz(self.dec_tree, out_file=None,
                                        # feature_names=list(train_set.columns),
                                        class_names=["0", "1"],
                                        filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        plt.show()

        # predict using the decision tree
        train_pred_dt = dec_tree.predict(train_set)
        test_pred_dt = dec_tree.predict(test_set)

        self.acc_dt_train, self.acc_dt_test = evaluation(train_label=train_label,
                                                         train_pred=train_pred_dt,
                                                         test_label=test_label,
                                                         test_pred=test_pred_dt)

        report_scores(train_label, train_pred_dt)  # training

        report_scores(test_label, test_pred_dt)  # test

        CrossValidation(dec_tree, train_set, train_label)

        tst_label = test_label.reset_index(drop=True).values

        disp = ConfusionMatrixDisplay.from_predictions(test_pred_dt, tst_label)

        disp.ax_.set_title('Decision Tree Confusion Matrix')
        plt.show()

    def saving_results(self, outputs):
        # Saving results
        key = "dt"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.dt_model_selection_time
        outputs[key]["params"] = self.dt_best_params
        outputs[key]["model"] = self.dec_tree
        outputs[key]["train_acc"] = self.acc_dt_train
        outputs[key]["test_acc"] = self.acc_dt_test

        return outputs

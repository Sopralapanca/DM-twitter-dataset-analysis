from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

from data_setup import train_label, test_set, test_label, train_set
import time
from utilities import random_search, evaluation, CrossValidation, report_scores


class NeuralNetwork:
    """Neural Network (Sklearn)"""

    def __init__(self, seed):
        self.acc_nn_test = None
        self.acc_nn_train = None
        self.nn_model_selection_time = None
        self.nn = None
        self.nn_best_params = None
        self.seed = seed
        layers = [1, 2, 3]
        units_per_layer = [16, 32, 64, 128]
        hidden_layer_sizes = []
        for units in units_per_layer:
            for layer in layers:
                tmp = []
                for i in range(layer):
                    tmp.append(units)
                hidden_layer_sizes.append(tmp)
        self.param_dist = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": [0.1, 0.001, 0.0001, 0.00001, 0.000001],
            "activation": ["logistic", "tanh", "relu"],
            "alpha": [0.1, 0.001, 0.0001, 0.00001],
            "learning_rate": ["constant", "invscaling", "adaptive"]}

        self.max_iter = 1024
        self.early_stopping = True
        self.validation_fraction = 0.1

    def model_selection(self):
        """Model Selection"""

        nn = MLPClassifier(max_iter=self.max_iter,
                           validation_fraction=self.validation_fraction,
                           early_stopping=self.early_stopping, random_state=self.seed)

        start_time = time.time()
        self.nn_best_params = random_search(model=nn, dict_params=self.param_dist, plot=True, train_label=train_label,
                                       train_set=train_set)
        end_time = time.time()
        self.nn_model_selection_time = end_time - start_time
        print("Model selection time", self.nn_model_selection_time, "sec")

        hidden_layer_sizes = self.nn_best_params["hidden_layer_sizes"]
        learning_rate_init = self.nn_best_params["learning_rate_init"]
        learning_rate = self.nn_best_params["learning_rate"]
        alpha = self.nn_best_params["alpha"]
        activation = self.nn_best_params["activation"]

        self.nn = MLPClassifier(max_iter=self.max_iter,
                                validation_fraction=self.validation_fraction,
                                early_stopping=self.early_stopping,
                                hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init,
                                learning_rate=learning_rate, alpha=alpha, activation=activation)

        self.nn.fit(train_set, train_label)

        print(nn.score(train_set, train_label))
        plt.plot(nn.loss_curve_, label="loss")
        plt.plot(nn.validation_scores_, label="accuracy")
        plt.legend()

        train_pred_nn = nn.predict(train_set)
        test_pred_nn = nn.predict(test_set)

        """### Evaluation"""

        self.acc_nn_train, self.acc_nn_test = evaluation(train_label=train_label,
                                                         train_pred=train_pred_nn,
                                                         test_label=test_label,
                                                         test_pred=test_pred_nn)

        report_scores(train_label, train_pred_nn)
        report_scores(test_label, test_pred_nn)
        CrossValidation(nn, train_set, train_label)

        disp = ConfusionMatrixDisplay.from_predictions(test_label, test_pred_nn)
        disp.ax_.set_title('Neural Network Confusion Matrix')
        plt.show()

    def saving_results(self, outputs):
        # Saving results
        key = "nn"
        outputs[key] = {}
        outputs[key]["model_selection_time"] = self.nn_model_selection_time
        outputs[key]["params"] = self.nn_best_params
        outputs[key]["model"] = self.nn
        outputs[key]["train_acc"] = self.acc_nn_train
        outputs[key]["test_acc"] = self.acc_nn_test

        return outputs

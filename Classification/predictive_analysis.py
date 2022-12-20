import time

import numpy as np
import requests
from sklearn import metrics

from data_setup import test_set, test_label
from Classifiers.random_forest import RandomForest
from Classifiers.k_nearest_neighbor import KNearestNeighbors
from Classifiers.ada_boost import AdaBoost
from Classifiers.decision_tree import DecisionTree
from Classifiers.neural_network import NeuralNetwork
from Classifiers.support_vector import SupportVectorClassifier
from Classifiers.naive_bayes import NaiveBayes

# Constants & Values
seed = 42
outputs = {}
very_start_time = time.time()
prepro_wellprinted = 'no preprocessing'

bot_token = "5947990353:AAGTl2uNdIFaudjVDx1g-RcZ4iumjluQFHo"
chat_id = "-1001817299141"
message = "[Classification] start"

url = "https://api.telegram.org/bot{}/sendMessage".format(bot_token)

payload = {
    "chat_id": chat_id,
    "text": message
}

response = requests.post(url, json=payload)

"""Decision Tree"""
dt = DecisionTree(seed=seed)
dt.model_selection()
outputs = dt.saving_results(outputs)

"""Random Forest"""
random_forest = RandomForest(seed=seed)
random_forest.model_selection()
outputs = random_forest.saving_results(outputs)

"""Ada Boost"""
ab = AdaBoost(seed=seed)
# Decision Tree
ab.dt_based(dec_tree=outputs['dt']['model'])
outputs = ab.dt_saving_results(outputs)
print(outputs['dt_abc'])
# Random Forest
ab.rf_based(rf=outputs['rf']['model'])
outputs = ab.rf_saving_results(outputs)

"""KNN"""
knn = KNearestNeighbors(seed=seed)
knn.model_selection()
outputs = knn.saving_results(outputs)

"""Neural Network"""
nn = NeuralNetwork(seed=seed)
nn.model_selection()
outputs = nn.saving_results(outputs)

"""Support Vector Machine"""
svc = SupportVectorClassifier(seed=seed)
svc.model_selection()
outputs = svc.saving_results(outputs)

"""Naive Bayes"""
nbc = NaiveBayes(seed=seed)
nbc.model_selection()
outputs = nbc.saving_results(outputs)

"""Parameter Selection & Fitting of Classifiers"""

# TODO
# rule-based classifier?
# altri ensamble methods? -> bagging
# gradient boosting

"""# Results"""

print(outputs)

test_accuracies = []
models = []

for model in outputs:
    test_accuracies.append(outputs[model]["test_acc"])
    models.append(model)
    print(model, outputs[model]["test_acc"])

best_model = models[np.argmax(test_accuracies)]
print("Our best classifier is *", best_model,
      "* with an accuracy of", round(np.max(test_accuracies), 6), "on test set.")

model = outputs[best_model]["model"]

test_set_predicted = model.predict(test_set)

print("On the test set, we obtain an accuracy of", round(metrics.accuracy_score(test_label, test_set_predicted), 6),
      "%")

bot_token = "5947990353:AAGTl2uNdIFaudjVDx1g-RcZ4iumjluQFHo"
chat_id = "-1001817299141"
message = "[Classification][GS con la pala]\nfinita in " + str((time.time() - very_start_time)) + "sec\n\n" + str(
    prepro_wellprinted) + "\n🏆Winner:" + str(outputs[best_model])

url = "https://api.telegram.org/bot{}/sendMessage".format(bot_token)

payload = {
    "chat_id": chat_id,
    "text": message
}

response = requests.post(url, json=payload)

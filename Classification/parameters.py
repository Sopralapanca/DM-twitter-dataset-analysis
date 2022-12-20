from data_setup import n_components

"""Params"""
max_depth = [2, 3, 5, 7, 10, 12, 14, 16, 18, None]
max_features_list = list(range(1, n_components + 1, 2))
min_sample_split_list = list(range(4, 60, 4))
min_sample_leaf_list = list(range(2, 50, 4))
criterion = ["entropy", "gini", "log_loss"]
class_weight = ['balanced', None, {0: 0.3, 1: 0.7}]


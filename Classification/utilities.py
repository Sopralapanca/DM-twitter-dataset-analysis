import statistics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate

n_iter_search = 20  # number of trials random search


def random_search(model, dict_params, train_set, train_label, cv=5, plot=False):
    rs = RandomizedSearchCV(model, param_distributions=dict_params,
                            n_iter=n_iter_search,
                            n_jobs=-1,
                            cv=cv,
                            scoring=make_scorer(accuracy_score),
                            verbose=3,
                            return_train_score=True)

    rs.fit(train_set, train_label)

    if plot:
        plots(rs, dict_params)

    print('Best setting parameters ', rs.cv_results_['params'][0])
    print('Mean and std of this setting ', rs.cv_results_['mean_test_score'][0],
          rs.cv_results_['std_test_score'][0])

    return rs.cv_results_['params'][0]


def plots(random_search, dict_params):
    df = pd.DataFrame(random_search.cv_results_)

    if "param_class_weight" in df.columns:
        df['param_class_weight'] = df['param_class_weight'].astype("string")

    if "param_hidden_layer_sizes" in df.columns:
        df['param_hidden_layer_sizes'] = df['param_hidden_layer_sizes'].astype("string")

    if "bootstrap" in df.columns:
        df['bootstrap'] = df['bootstrap'].astype("string")

    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score',
               'std_train_score']

    # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
    def pooled_var(stds):
        n = 5  # size of each group
        return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))

    fig, axes = plt.subplots(1, len(dict_params),
                             figsize=(5 * len(dict_params), 7),
                             sharey='row')
    if len(dict_params) > 1:
        axes[0].set_ylabel("Score", fontsize=25)
    else:
        axes.set_ylabel("Score", fontsize=25)

    lw = 2

    for idx, (param_name, param_range) in enumerate(dict_params.items()):

        grouped_df = df.groupby(f'param_{param_name}')[results] \
            .agg({'mean_train_score': 'mean',
                  'mean_test_score': 'mean',
                  'std_train_score': pooled_var,
                  'std_test_score': pooled_var})

        previous_group = df.groupby(f'param_{param_name}')[results]
        if len(dict_params) > 1:
            axes[idx].set_xlabel(param_name, fontsize=30)
            axes[idx].set_ylim(0.0, 1.1)

            x_values = grouped_df['mean_train_score'].axes[0]

            axes[idx].plot(x_values, grouped_df['mean_train_score'], label="Training score",
                           color="darkorange", lw=lw)

            if param_name == "bootstrap":
                x_values = x_values.astype("str")

            # axes[idx].fill_between(x_values, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
            #                        grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
            #                        color="darkorange", lw=lw)

            axes[idx].plot(x_values, grouped_df['mean_test_score'], label="Cross-validation score",
                           color="navy", lw=lw)

            # axes[idx].fill_between(x_values, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
            #                        grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
            #                        color="navy", lw=lw)
        else:
            axes.set_xlabel(param_name, fontsize=30)
            axes.set_ylim(0.0, 1.1)

            x_values = grouped_df['mean_train_score']

            axes.plot(x_values, grouped_df['mean_train_score'], label="Training score",
                      color="darkorange", lw=lw)

            if param_name == "bootstrap":
                x_values = x_values.astype("str")

            # axes.fill_between(x_values, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
            #                   grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
            #                   color="darkorange", lw=lw)

            axes.plot(x_values, grouped_df['mean_test_score'], label="Cross-validation score",
                      color="navy", lw=lw)

            # axes.fill_between(x_values, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
            #                   grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
            #                   color="navy", lw=lw)

    if len(dict_params) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles, labels = axes.get_legend_handles_labels()

    fig.suptitle('Validation curves', fontsize=40)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    plt.show()


# Grid search with Cross Validation

def grid_search(model, dict_params, train_set, train_label, cv=5, verbose=False, plot=False):
    gs = GridSearchCV(model, dict_params, cv=cv, return_train_score=True)
    gs.fit(train_set, train_label)

    if plot:
        plots(gs, dict_params)

    if verbose:
        print(gs.cv_results_)
    return gs.best_params_


"""## Evaluation utilities"""


def report_scores(test_label, test_pred):
    print(classification_report(test_label, test_pred,
                                target_names=["Non-Bot", "Bot"]))


# evaluate the accuracy on the train set and the test set
# metrics also contains precision, recall, f1 and the support

def evaluation(train_label, train_pred, test_label, test_pred):
    acc_train = metrics.accuracy_score(train_label, train_pred)
    print('Accuracy train set ', acc_train)
    acc_test = metrics.accuracy_score(test_label, test_pred)
    print('Accuracy test set ', acc_test)
    print()
    print('Precision train set ', metrics.precision_score(train_label, train_pred, average='weighted'))
    print('Recall train set ', metrics.recall_score(train_label, train_pred, average='weighted'))
    print('F1 score train set ', metrics.f1_score(train_label, train_pred, average='weighted'))
    print('Support train set ', metrics.precision_recall_fscore_support(train_label, train_pred))
    print()
    print('Precision test set ', metrics.precision_score(test_label, test_pred, average='weighted'))
    print('Recall test set ', metrics.recall_score(test_label, test_pred, average='weighted'))
    print('F1 score test set ', metrics.f1_score(test_label, test_pred, average='weighted'))
    print('Support test set ', metrics.precision_recall_fscore_support(test_label, test_pred))
    return acc_train, acc_test


# cross-validation to better evaluate the model

def CrossValidation(model, train_set, train_label):
    scores = cross_validate(model, train_set, train_label, cv=3, return_train_score=True)
    print('Fit time ', statistics.mean(scores['fit_time']))
    print('Score time ', statistics.mean(scores['score_time']))
    print('Test score ', statistics.mean(scores['test_score']))
    print('Train score ', statistics.mean(scores['train_score']))


def discretize_data(dataset, variables):
    for variable in variables:
        # get the unique variable's values
        var = sorted(dataset[variable].unique())

        # generate a mapping from the variable's values to the number representation
        mapping = dict(zip(var, range(0, len(var) + 1)))

        # add a new colum with the number representation of the variable
        dataset[variable + '_num'] = dataset[variable].map(mapping).astype(int)
    return dataset

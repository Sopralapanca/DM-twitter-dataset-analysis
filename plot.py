from main import tweets_df, users_df
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import calendar


def histogram_tweets_periods(df, hue=None, x_col=None, multiple_mode=None,
                             xlabel=None, legend=None, range=None, tick_labels=None):

    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 8))
    sns.despine(f)

    g = sns.histplot(
        df,
        x=x_col, hue=hue,
        multiple=multiple_mode,
        palette=sns.color_palette("husl", 13),
        log_scale=False,
        discrete=True,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0.8, 0.9), title=legend)

    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks(range)

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(tick_labels,
                       fontdict={'horizontalalignment': 'right', 'fontsize': 12, 'rotation': 30})
    ax.set_xlabel(xlabel)
    plt.show()


def boxplot_tweet_dimension(df, y_col=None, x_col=None, ylabel=None, xlabel=None):
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    sns.set_theme(style="ticks", palette=sns.color_palette("husl", 2))

    # Plot the box for quartiles
    sns.boxplot(x=x_col, data=df,
                whis=[0, 100], width=.4)

    # Add the points representing each tweet text length
    sns.stripplot(x=x_col, data=tweet_dim,
                  size=4, linewidth=0, alpha=0.4)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="", xlabel=xlabel)
    sns.despine(trim=True, left=True)
    ax.tick_params(left=False)
    plt.show()


tweet_dim = pd.DataFrame(columns=['text'])
tweet_dim['text'] = tweets_df['text'].str.len()

boxplot_tweet_dimension(tweet_dim, None, "text", None, "Length per character")

histogram_tweets_periods(tweets_df, tweets_df['created_at'].dt.year, tweets_df['created_at'].dt.month, 'layer',
                         'Months', 'Years', range(100, 1000, 100), [month for month in calendar.month_name[1:]])

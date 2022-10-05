from main import tweets_df, users_df
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import calendar


def histogram_tweets_periods():
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 8))
    sns.despine(f)

    print(tweets_df.head(20)['created_at'])

    g = sns.histplot(
        tweets_df,
        x=tweets_df['created_at'].dt.month, hue=tweets_df['created_at'].dt.year,
        multiple='layer',
        palette=sns.color_palette("husl", 13),
        log_scale=False,
        discrete=True,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0.8, 0.9), title='Years')

    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks(range(100, 1000, 100))

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels([month for month in calendar.month_name[1:]],
                       fontdict={'horizontalalignment': 'right', 'fontsize': 12, 'rotation': 30})
    ax.set_xlabel('Months')
    plt.show()


def boxplot_tweet_dimension():
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))

    sns.set_theme(style="ticks", palette=sns.color_palette("husl", 2))
    tweet_dim = pd.DataFrame(columns=['text'])
    tweet_dim['text'] = tweets_df['text'].str.len()

    # Plot the box for quartiles
    sns.boxplot(x="text", data=tweet_dim,
                whis=[0, 100], width=.4)

    # Add the points representing each tweet text length
    sns.stripplot(x="text", data=tweet_dim,
                  size=4, linewidth=0, alpha=0.4)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="", xlabel="Length per character")
    sns.despine(trim=True, left=True)
    ax.tick_params(left=False)
    plt.show()


boxplot_tweet_dimension()

import pandas as pd
import seaborn as sns
import calendar
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

from main import tweets_df


# Prova per rappresentare i dati in heatmap, ma i dati sono troppo sparsi
def sb_heatmap(df):
    # Load the example flights dataset and convert to long-form
    flights_long = sns.load_dataset("flights")
    print(flights_long.info())
    print(flights_long.head(10))
    flights = flights_long.pivot("month", "year", "passengers")

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

    mapping = {index: month for index, month in enumerate(calendar.month_abbr) if month}
    # Load the example flights dataset and convert to long-form
    tweets = pd.DataFrame(columns=['year', 'month', "value"])
    tweets['year'] = tweets_df['created_at'].dt.year.unique()
    tweets['month'] = tweets_df['created_at'].dt.month
    # tweets["month"].replace(mapping, inplace=True)
    tweets['value'] = tweets_df['created_at'].dt.month.value_counts().astype(int)
    tweets['value'] = tweets['value'].fillna(0)
    tweets['value'] = tweets['value'].astype(int)

    tweets = tweets.pivot("month", "year", "value")
    tweets = tweets.fillna(0)

    print(tweets.info())
    print(tweets.head(13))

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(tweets, annot=True, linewidths=.5, ax=ax)
    plt.show()


def sb_histogram(df, hue=None, x_col=None, calendar_view=False, range=None):
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 8))
    sns.despine(f)

    # Create the histogram setting the column to be represented and the one to overlap
    g = sns.histplot(
        df,
        x=x_col, hue=hue,
        multiple='layer',
        palette=sns.color_palette("husl", 13),
        log_scale=False,
        discrete=True,
    )

    # Tweak the visual presentation
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks(range)
    if calendar_view:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax.set_xlabel('Months')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(0.8, 0.9), title='Year')
        ax.set_xticklabels([month for month in calendar.month_name[1:]],
                           fontdict={'horizontalalignment': 'right', 'fontsize': 12, 'rotation': 30})
    plt.show()


def sb_boxplot(df, y_col="", x_col="", ylabel="", xlabel="", range=None):
    f, ax = plt.subplots(figsize=(7, 6))
    sns.set_theme(style="ticks", palette=sns.color_palette("husl", 2))

    # Plot the box for quartiles
    sns.boxplot(x=x_col, data=df,
                whis=[5, 95], width=.3)

    # Add the points representing each tweet text length
    sns.stripplot(x=x_col, data=df,
                  size=4, linewidth=0, alpha=0.4)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="", xlabel=xlabel)
    sns.despine(trim=True, left=True)
    ax.tick_params(left=False)
    ax.set_xticks(range)
    plt.show()


# Histogram showing the number of tweets made per year overlapping years on the same months
sb_histogram(tweets_df, tweets_df['created_at'].dt.year, tweets_df['created_at'].dt.month, True, range(100, 1100, 100))

# As we can see in this box plot we have 90% of the tweets in b/w about the 20 and 140 characters
tweet_dim = pd.DataFrame(columns=['text'])
tweet_dim['text'] = tweets_df['text'].str.len()
sb_boxplot(tweet_dim, "", "text", "", "Tweets lenght", range(0, 260, 20))

print(tweets_df.info())
print(tweets_df['favorite_count'].value_counts())

# Here we can see that the majority of tweets are getting 0 like
sb_histogram(tweets_df, None, "favorite_count", False, range(0, 5000, 200))

# sb_heatmap(tweets_df)

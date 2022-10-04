# USERS CSV
#
# 1. User Id: a unique identifier of the user
# 2. Statues Count: the count of the tweets made by the user at the moment of data
# crawling
# 3. Lang: the user’s language selected
# 4. Created at: the timestamp in which the profile was created
# 5. Label: a binary variable that indicates if a user is a bot or a genuine user
#
# TWEETS CSV
#
# 1. ID: a unique identifier for the tweet
# 2. User Id: a unique identifier for the user who wrote the tweet
# 3. Retweet count: number of retweets for the tweet in analysis
# 4. Reply count: number of reply for the tweet in analysis
# 5. Favorite count: number of favorites (likes) received by the tweet
# 6. Num hashtags: number of hashtags used in the tweet
# 7. Num urls: number of urls in the tweet
# 8. Num mentions: number of mentions in the tweet
# 9. Created at: when the tweet was created
# 10. Text: the text of the tweet

import pandas as pd
from utils.tools_ntbk import find_and_remove_duplicates

max_rows = 200000

tweets_df = pd.read_csv("./data/tweets.csv", nrows=max_rows)
users_df = pd.read_csv("./data/users.csv", nrows=max_rows)

# Print info about the dataset

print(tweets_df.info())
print(users_df.info())
print(users_df.isna().any())
print(tweets_df.isna().any())

# As shown above there are some null values inside the two dataframes. Furthermore, the type of the column values is
# 'object' even for columns that should only have numeric values such as 'id', 'user_id' and others. This means that
# some values in the respective columns are not integers but strings or something else. The data is therefore to be
# cleaned.

print(tweets_df.head(5))
print(users_df.head(5))

# Checking NaN and Duplicates

# Duplicates
users_df, count1 = find_and_remove_duplicates(users_df)
print("duplicates removed from user.csv file \t", count1)

original_lenght_tweet = len(tweets_df)
print("Len of df_tweet before cleaning", original_lenght_tweet)
tweets_df, count_tweet_duplicated_removed = find_and_remove_duplicates(tweets_df)
print("duplicates removed from tweets.csv file: \t", count_tweet_duplicated_removed)
print("Len of df_tweet after cleaning", len(tweets_df))
print("We have ", round((count_tweet_duplicated_removed / original_lenght_tweet), 4),
      "% of duplicates in tweets.csv file")

# checking NaN
print("\nChecking NaN in Tweets Dataframe")
for col in tweets_df.columns:
    print(f"column: {col} NaN: {tweets_df[col].isna().sum()}")

print("\nChecking NaN in Users Dataframe")
for col in users_df.columns:
    print(f"column: {col} NaN: {users_df[col].isna().sum()}")


# Assign correct type to attributes
# covert tweets_df columns from object to numeric. If a value can't be converted to integer a NaN is inserted

columns = ["id", "user_id", "retweet_count", "reply_count",
           "favorite_count", "num_hashtags", "num_urls", "num_mentions"]

for col in columns:
    tweets_df[col] = pd.to_numeric(tweets_df[col], errors='coerce', downcast='integer')

tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors='coerce', yearfirst=True)
print(tweets_df.info())

users_df["created_at"] = pd.to_datetime(users_df["created_at"], errors='coerce', yearfirst=True)
print(users_df.info())
print(tweets_df.info())




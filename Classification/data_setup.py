import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utilities import discretize_data

df = pd.read_csv("users_new.csv", index_col='user_id')
df['name'] = df['name'].fillna("UNKNOWN")
df.drop(columns=['Unnamed: 0', "min_delta_sec"], inplace=True)
variables = ['lang']
df = discretize_data(df, variables)
label = df.pop('bot')
df.drop(columns=['name', 'lang', 'subscription_date'], axis=1, inplace=True)

print("We have a dataset of", len(label), "users.")
print("The", round(label.value_counts()[1] / len(label), 4) * 100, "% are bots, while the",
      round(label.value_counts()[0] / len(label), 4) * 100, "% are genunine users.")

test_size = 0.10  # test set size

"""# Scaling the data"""
for column in df.columns:
    df[column] = df[column].apply(lambda x: np.log(x + 1))

scaler = StandardScaler()
scaler.fit(df.values)
scaled_values = scaler.transform(df.values)

df_scaled = pd.DataFrame(scaled_values, columns=df.columns)
train_set, test_set, train_label, test_label = train_test_split(df_scaled,
                                                                label, shuffle=True, stratify=label,
                                                                test_size=test_size)
n_components = len(train_set.columns)


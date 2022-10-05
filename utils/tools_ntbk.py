import gdown
from pandas import DataFrame

USERS = "16b7Gbe69-SaVzGc1x3s3s7b8lzHgd5BR"
TWEETS = "1qYoicySRBbLi9Y8ZytMEi9ee2dIhxxUe"


def download_dataset():  # Download dataset from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={USERS}", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={TWEETS}", quiet=False)


def find_and_remove_duplicates(df: DataFrame):
    original_lenght = len(df)
    df = df.drop_duplicates()
    new_lenght = len(df)
    return df, original_lenght - new_lenght

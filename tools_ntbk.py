import gdown

USERS = "16b7Gbe69-SaVzGc1x3s3s7b8lzHgd5BR"
TWEETS = "1qYoicySRBbLi9Y8ZytMEi9ee2dIhxxUe"


def download_dataset():     # Download dataset from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={USERS}", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={TWEETS}", quiet=False)

import kagglehub
import os

def download():
    dataset = "deasadiqbal/private-data-1"

    print("Downloading dataset:", dataset)

    # by default it is under ~/.cache/kagglehub/datasets
    # for caching purposes.
    # path is returned.
    path = kagglehub.dataset_download(dataset)

    print("Dataset downloaded to:", path)

if __name__ == "__main__":
    download()

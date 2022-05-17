import math
import tarfile
import zipfile
import pandas as pd
import urllib.request
import os, shutil
import ijson

# Get current folder
current_path = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(current_path, "datasets")
clean_folder = os.path.join(current_path, "cleaned-data")


def download_dataset(url, file_name):
    urllib.request.urlretrieve(
        url,
        os.path.join(dataset_folder, file_name),
        reporthook=(
            lambda count, block, total: print(
                f"Downloading {file_name}: {math.floor((count * block) / total * 100)}%",
                end="\r",
            )
        ),
    )
    print(f"Downloaded {file_name} from {url}")


# Check if the dataset already exists
if not (os.path.exists(dataset_folder) and os.path.exists(clean_folder)):
    shutil.rmtree(dataset_folder)
    shutil.rmtree(clean_folder)
    os.mkdir(dataset_folder)
    os.mkdir(clean_folder)

    download_dataset("https://data.deepai.org/squad1.1.zip", "squad.zip")
    download_dataset(
        "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz", "race.tar.gz"
    )

    with zipfile.ZipFile(os.path.join(dataset_folder, "squad.zip"), mode="r") as obj:
        obj.extractall(os.path.join(dataset_folder, "squad/"))

    with tarfile.open(os.path.join(dataset_folder, "race.tar.gz"), mode="r") as obj:
        obj.extractall(os.path.join(dataset_folder, "race/"))

# This list will store all the Q&A
source_list = []
target_list = []

# Start cleaning data
squad = open(os.path.join(dataset_folder, "squad/train-v1.1.json"))
objects = ijson.items(squad, "data.item")

for obj in objects:
    title = obj["title"]
    paragraphs = obj["paragraphs"]
    for p in paragraphs:
        context = p["context"]
        qas = [p for p in p["qas"] if len(p) > 0]

        source_text = f"generate {len(qas)} questions: {context}"
        target_text = ""

        for number, qa in enumerate(qas):
            target_text += (
                f"{number + 1}. {qa['question']}\nA: {qa['answers'][0]['text']}\n"
            )

        source_list.append(source_text)
        target_list.append(target_text)

dataframe = pd.DataFrame({"source_text": source_list, "target_text": target_list})

train = dataframe.sample(frac=0.8, random_state=20)
test = dataframe.drop(train.index)

dataframe.to_csv(os.path.join(clean_folder, "squad.pkl"))
train.to_csv(os.path.join(clean_folder, "squad-train.pkl"))
test.to_csv(os.path.join(clean_folder, "squad_test.pkl"))

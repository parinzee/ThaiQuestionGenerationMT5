import ijson
import pandas as pd


f = open("dataset/squad-v2.0.json")
objects = ijson.items(f, "data.item")

source_list = []
target_list = []

for obj in objects:
    title = obj["title"]
    paragraphs = obj["paragraphs"]
    for p in paragraphs:
        context = p["context"]
        qas = [p for p in p["qas"] if not p["is_impossible"]]

        if len(qas) > 0:
            source_text = f"generate {len(qas)} questions: {context}"
            target_text = ""

            for number, qa in enumerate(qas):
                target_text += f"{number + 1}. {qa['question']}\nA: {qa['answers'][0]['text']}\n"

            source_list.append(source_text)
            target_list.append(target_text)

dataframe = pd.DataFrame(
    {"source_text": source_list, "target_text": target_list})

train = dataframe.sample(frac=0.8, random_state=20)
test = dataframe.drop(train.index)

dataframe.to_csv("cleaned-data/dataset.csv")
train.to_csv("cleaned-data/train.csv")
test.to_csv("cleaned-data/test.csv")

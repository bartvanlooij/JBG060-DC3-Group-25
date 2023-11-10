import multiprocessing
from transformers import pipeline, AutoTokenizer
from multiprocessing import Process
import numpy as np
import pandas as pd

df_full = pd.read_csv("data/full_dataset_with_sentiment.csv", index_col=0)
from tqdm import tqdm

tqdm.pandas()


def get_sentiment_roberta_split(df_section, start, end):
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding="max_length", max_length=512, truncation=True
    )
    sentiment_task = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
    )
    df_section["sentiment_roberta_sum"] = df_section["summary"].progress_apply(
        lambda x: get_sentiment_roberta(x, sentiment_task)
    )
    df_section["sentiment_roberta"] = df_section["paragraphs"].progress_apply(
        lambda x: get_sentiment_roberta(x, sentiment_task)
    )
    df_section.to_csv(
        f"data/full_dataset_with_sentiment_{start}_{end}_sum.csv", index=True
    )


def get_sentiment_roberta(row, sentiment_task):
    sent = sentiment_task(row)
    return sent[0]["label"]


if __name__ == "__main__":  # confirms that the code is under main function
    # create a list of indexes to split the dataframe into 10 parts
    df_split = np.array_split(df_full, 10)
    df_split = [df_section.copy() for df_section in df_split]
    # create processes equal to the number of parts
    processes = []
    for i in range(8):
        print("Starting process {}...".format(i))
        p = multiprocessing.Process(
            target=get_sentiment_roberta_split,
            args=(df_split[i], df_split[i].index[0], df_split[i].index[-1]),
        )
        processes.append(p)
        p.start()
    # complete the processes
    for proc in processes:
        proc.join()

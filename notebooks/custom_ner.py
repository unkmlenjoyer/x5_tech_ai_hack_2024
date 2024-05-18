"""Script for customize NER"""

# %%
import json
from typing import Dict

import razdel
from config import HackConfig
from datasets import DatasetDict, Dataset

from collections import Counter
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from transformers import AutoModelForTokenClassification, AutoTokenizer

# %%
conf = HackConfig()

# %%
with open(conf.path_raw_train_data, "r") as f:
    raw_data = json.load(f)


# %%
def prepare_item(item: Dict) -> Dict:
    """Preprocess raw sample to B-I-O marks

    Parameters
    ----------
    item : Dict
        Sample of data contains text, entities. Each entity has text, coords, group

    Returns
    -------
    Dict
        Tokens and token labels
    """
    raw_tokens = list(razdel.tokenize(item["text"]))
    tokens = [el.text for el in raw_tokens]
    token_labels = ["O"] * len(tokens)
    chars = [None] * len(item["text"])
    for i, word in enumerate(raw_tokens):
        chars[word.start : word.stop] = [i] * len(word.text)

    # map entity tags to tokens
    for entity in item["entities"]:
        # extract tokens id for current entity
        entity_tokens = sorted(
            {idx for idx in chars[entity["start"] : entity["end"]] if idx is not None}
        )

        # first entity token is B-ENTITY_GROUP
        token_labels[entity_tokens[0]] = "B-" + entity["entity_group"]

        # others entity tokens should be I-ENTITY_GROUP
        for e_t in entity_tokens[1:]:
            token_labels[e_t] = "I-" + entity["entity_group"]

    return {"tokens": tokens, "tags": token_labels}


# %%
expanded_data = np.array([prepare_item(item) for item in raw_data])

# %%
labels = sorted({label for item in expanded_data for label in item["tags"]})
if "O" in labels:
    labels.remove("O")
    labels = ["O"] + labels

# %%
labels_count = (
    pd.DataFrame([dict(Counter(item["tags"]).items()) for item in expanded_data])
    .fillna(0)
    .astype(int)
    .values
)

# %%
row_ids = np.arange(labels_count.shape[0])

train_idx, _, test_idx, _ = iterative_train_test_split(
    row_ids[:, np.newaxis], labels_count, test_size=0.3
)


# %%
splitted = DatasetDict(
    {
        "train": Dataset.from_pandas(
            pd.DataFrame().from_records(expanded_data[train_idx.reshape(-1)])
        ),
        "test": Dataset.from_pandas(
            pd.DataFrame().from_records(expanded_data[test_idx.reshape(-1)])
        ),
    }
)

# %%
model = AutoModelForTokenClassification.from_pretrained(conf.base_model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(conf.base_model_checkpoint)

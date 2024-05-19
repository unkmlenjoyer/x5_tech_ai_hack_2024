"""Script for customize NER"""

# %%
import json
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import razdel
from config import HackConfig
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

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
id2label = dict(enumerate(labels))
label2id = {label: idx for idx, label in id2label.items()}

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
    row_ids[:, np.newaxis], labels_count, test_size=conf.test_size
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
tokenizer = AutoTokenizer.from_pretrained(conf.base_model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# %%
def preprocessing_ner_labels(samples, label_all_tokens=True):
    tokenizer_out = tokenizer(
        samples["tokens"], is_split_into_words=True, truncation=True
    )
    preprocessed_labels = []

    for i, tags in enumerate(samples["tags"]):
        word_ids = tokenizer_out.word_ids(batch_index=i)

        prev_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(tags[word_idx] if label_all_tokens else -100)

            prev_word_idx = word_idx

        label_ids = [
            label2id[idx] if isinstance(idx, str) else idx for idx in label_ids
        ]

        preprocessed_labels.append(label_ids)

    tokenizer_out["labels"] = preprocessed_labels
    return tokenizer_out


# %%
tokenized_datasets = splitted.map(
    preprocessing_ner_labels,
    batched=True,
    remove_columns=splitted["train"].column_names,
)


# %%
def compute_metrics(pred_input):
    predictions, labels = pred_input
    predictions = np.argmax(predictions, axis=2)

    y_pred = [
        [id2label[p] for p, lp in zip(pred, label) if lp != -100]
        for pred, label in zip(predictions, labels)
    ]

    y_true = [
        [id2label[lp] for p, lp in zip(pred, label) if lp != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        digits=2,
        suffix=False,
        output_dict=True,
        mode=None,
        sample_weight=None,
        zero_division=1,
        scheme=None,
    )

    return results["macro avg"]


# %%
model = AutoModelForTokenClassification.from_pretrained(
    conf.base_model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

tr_args = TrainingArguments(
    output_dir="rubert_tiny2_v1",
    learning_rate=4e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    weight_decay=0.3,
    eval_strategy="epoch",
    save_strategy="no",
    push_to_hub=False,
    logging_steps=10,
    seed=conf.random_seed,
    data_seed=conf.random_seed,
)

trainer = Trainer(
    model=model,
    args=tr_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# %%
train_result = trainer.train()

# %%
trainer.evaluate(tokenized_datasets["train"])

# %%
trainer.evaluate(tokenized_datasets["test"])
# %%
text = "**Критикуйте конструктивно, а не публично.**\n\nЕсли у вас есть вопросы или предложения по улучшению работы компании, вы можете поговорить об этом со своим руководителем, написать на электронный ящик горячей линии krasilnikovamos@gruppa.edu или оставить комментарий на [корпоративном портале.](http://rao.org/)\n\n**Воздержитесь от критики конкурентов.**"
# %%
pipe = pipeline(
    model=model, tokenizer=tokenizer, task="ner", aggregation_strategy="average"
)
# %%
pipe(text)

# %%
model.save_pretrained("models/rubert_tiny2_v1")
tokenizer.save_pretrained("models/rubert_tiny2_v1")

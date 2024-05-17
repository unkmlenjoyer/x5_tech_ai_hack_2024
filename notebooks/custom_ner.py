"""Script for customize NER"""

# %%
import json

from config import HackConfig

# %%
conf = HackConfig()

# %%
with open(conf.path_raw_train_data, "r") as f:
    train_data = json.load(f)

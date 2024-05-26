"""Generate predictions for hack"""

import json

import torch
from transformers import pipeline

device = torch.device(
    "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
)

# Load the fine-tuned model
model = pipeline(
    "ner",
    model="../models/dp_v1_ft",
    tokenizer="../models/dp_v1_ft",
    aggregation_strategy="average",
    device=device,
)

with open("../data/test.json", "r") as file:
    data = json.load(file)

# Make predictions on the test set and format the results
predictions_output = []
for item in data:
    text = item["text"]
    entities = [
        {"entity_group": e["entity_group"], "start": e["start"], "end": e["end"]}
        for e in model(text)
    ]

    predictions_output.append({"text": text, "entities": entities})

# Save the predictions_output to a JSON file with UTF-8 encoding
output_file = "../data/submission.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(predictions_output, f, ensure_ascii=False, indent=4)

print(f"Predictions saved to {output_file}")

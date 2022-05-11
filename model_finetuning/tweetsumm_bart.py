# !pip install datasets
# !pip install sentencepiece
# !pip install transformers
# !pip install rouge_score
# !pip install bert_score 

from bert_score import score
from datasets import load_dataset, Dataset, load_metric
import sys
from google.colab import drive
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import huggingface_hub
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt")
import gc
from torch import nn 

# ----- Check if GPU is connected ----- # 
gpu_info = nvidia-smi -L
gpu_info = "\n".join(gpu_info)
if gpu_info.find("failed") >= 0:
    print("Not connected to a GPU")
else:
    print(gpu_info)

# ----- Mounting Google Drive ----- # 

drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/CIS6930_final')

# ----- Importing TweetSum processing module ----- #
from tweet_sum_processor import TweetSumProcessor

# ----- Torch Device ----- #
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------

# --- DEFINE MODEL AND TOKENIZER --- #
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# ----- Metric
metric = load_metric("rouge")

# ---- Freeze parameters

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for BART and PEGASUS, just token embeddings for t5."""
    model_type = model.config.model_type
    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

freeze_embeds(model)

# ----- Reading in the Dataset
raw_datasets = load_dataset('csv', data_files={'train': '/content/drive/MyDrive/CIS6930_final/tweetsum_train.csv',
                                          'valid': '/content/drive/MyDrive/CIS6930_final/tweetsum_valid.csv',
                                          'test': '/content/drive/MyDrive/CIS6930_final/tweetsum_test.csv'})


max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    model_inputs = tokenizer(examples["inputs"], max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summaries"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

batch_size = 1
args = Seq2SeqTrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    output_dir = '/content/drive/MyDrive/CIS6930_final/results/bart', 
    logging_dir = '/content/drive/MyDrive/CIS6930_final/logs/bart'
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

# --------------------- # 
#    TEST EVALUATION    #
# --------------------- #

out = trainer.predict(tokenized_datasets["test"])
generated_summaries = []
for i in range(0, 110): 
  generated_summaries.append(tokenizer.decode(out[0][i], skip_special_tokens =  True))
ground_truth = tokenized_datasets["test"]["summaries"]
conversation = tokenized_datasets["test"]["inputs"]

print(out.metrics)

P, R, F1 = score(generated_summaries, ground_truth, lang="en", verbose=True)

print(P,R,F1)

print(f"System level F1 score: {F1.mean():.3f}")
print(f"System level precision score: {P.mean():.3f}")
print(f"System level recall score: {R.mean():.3f}")

# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/CIS6930_final')
# bart_summaries = pd.DataFrame({"candidate": generated_summaries, "reference": ground_truth, "conversation": conversation})
# bart_summaries.to_csv('/content/drive/MyDrive/CIS6930_final/summaries/bart_test_summaries2.csv') 
# print("Done")
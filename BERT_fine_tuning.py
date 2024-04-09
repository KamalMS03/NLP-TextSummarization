#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import torch
import pandas as pd
import logging
logging.disable(logging.WARNING)


# In[2]:


news_dataset_cnndailymail = pd.read_parquet( "/kaggle/input/cnn-dailymail/train-00000-of-00003.parquet")
news_dataset_cnndailymail.drop(columns=["id"], inplace=True )
news_dataset_cnndailymail.rename(columns={"highlights":"summary"},inplace=True)
 news_dataset_cnndailymail = news_dataset_cnndailymail[0:100]
rows_count = news_dataset_cnndailymail.shape[0]
news_dataset_cnndailymail


# In[3]:


import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset


# In[4]:


news_dataset_cnndailymail = Dataset(pa.Table.from_pandas(news_dataset_cnndailymail))
news_dataset_cnndailymail = news_dataset_cnndailymail.train_test_split(test_size=0.1)
news_dataset_cnndailymail


# In[5]:


tokenizer = BertTokenizer.from_pretrained("facebook/bert-large-cnn")
model = BertForSequenceClassification.from_pretrained("facebook/bert-large-cnn")


# In[6]:


prefix = "summarize: "

def preprocess_function( data ):
    inputs = [prefix + doc for doc in data["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    labels = tokenizer(text_target=data["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_news_dataset_cnndailymail = news_dataset_cnndailymail.map(preprocess_function,batched = True)


# In[7]:


tokenized_news_dataset_cnndailymail


# In[19]:


tokenized_train = tokenized_news_dataset_cnndailymail['train'].remove_columns(["article", "summary", "labels"])
tokenized_train.set_format("torch")
tokenized_train.column_names


# In[20]:


tokenized_eval = tokenized_news_dataset_cnndailymail['test'].remove_columns(["article", "summary", "labels"])
tokenized_eval.set_format("torch")
tokenized_eval.column_names


# In[21]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[22]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_train, shuffle=True, batch_size=4, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_eval, batch_size=4, collate_fn=data_collator
)


# In[23]:


train_dataloader


# In[24]:


import numpy as np
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


# In[25]:


outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)


# In[33]:


from transformers import create_optimizer, AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)


# In[34]:


from transformers import get_scheduler

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


# In[17]:


import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device


# In[32]:


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        print(loss)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# In[ ]:


trainer.train()
model.save_pretrained("./fine_tuned_bert")


# In[ ]:





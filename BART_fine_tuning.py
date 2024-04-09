#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
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
# news_dataset_cnndailymail = news_dataset_cnndailymail[0:100]
rows_count = news_dataset_cnndailymail.shape[0]
news_dataset_cnndailymail


# In[3]:


get_ipython().system('pip install datasets==2.15')
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset


# In[4]:


news_dataset_cnndailymail = Dataset(pa.Table.from_pandas(news_dataset_cnndailymail))
news_dataset_cnndailymail = news_dataset_cnndailymail.train_test_split(test_size=0.1)
news_dataset_cnndailymail


# In[5]:


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large-cnn")


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
model.save_pretrained("./fine_tuned_bart")


# In[ ]:
input_text = "Curious how cutting edge research is addressing complex issues? Then check out If Then, the new podcast where Stanford Graduate School of Business professors share the innovations they're most excited about from AI to sustainability and power. Listen to If Then wherever you get your podcasts. Welcome to the HBR IdeaCast from Harvard Business Review. I'm Alison Beard. When you hear the word friction, where does your mind go? Do you think of it as a force that leads to something positive and useful, like how rubbing two sticks together can create a fire? Or do you think of it as an enemy of progress, a slowdown mechanism, bureaucracy, conflict? Our guests today have spent the past seven years investigating friction in organizations. The good kind that can lead to wiser decisions and more innovative solutions, and the bad kind that leads to inefficiency and waste, exasperated employees and unhappy customers and clients. They say that all of us, no matter our level, need to better recognize when friction is helping or hindering our success, and learn how to add or subtract it as necessary. Bob Sutton is a professor emeritus at Stanford University and co-founder of its Center for Work, Technology and Organization and its D.School and Technology Ventures program. Huggy Rouse, professor of organizational behavior, human resources and sociology at Stanford, and the director of its TELUS leadership forum. They co-authored the book The Friction Project, how smart leaders make the right things easier and the wrong things harder, as well as the HBR article, Read Your Organization of Obstacles That Infuriate Everyone. Bob, Huggy, thanks so much for joining me. It's great to talk. A delight to be here. Thank you so much, Allison. So what got you two thinking together about this idea of friction as both a force for good and a force for evil? Bob and I wrote an earlier book called Scaling Up Excellence. And as we were sharing, we found that the top and senior echelons of an enterprise, they were very drawn to the message. But as we went lower down in the organization, employees lamented about how hard it was to get anything done in the first place. Let me give you two bookends of responses. We asked one executive a very simple question. Where do you work? And the guy looks at us with a glint in his eye and he says, I work in a frustration factory. That startled us. The other bookend was a young woman. I can never forget her. And she said, I spend most of my day putting myself into doing BS inconsequential work in my company. And when I go home, all I've got left are the scraps of myself for my family. That was like a hit in the solar plexus for both of us. We knew people and did even some consulting for Facebook, Salesforce, and Google. And their dream was to build a giant company. Well, they built giant companies, but man, is it hard to get things done in those places. That's just one of the problems when organizations get large and complex. It's harder to get things done. A lot of our work moved from focusing on how to scale organizations to how to deal with size and complexities. As we went through the journey, we realized friction is kind of like two-sided. Just as you've got bad bacteria in our microbiome, we have bad friction that infuriates people. But just as we have good bacteria in our biome, good friction actually helps people. What bad friction does is it makes it very hard for employees to choose a more curious and generous version of themselves. You're just overwhelmed by all of this. It's super hard for you to be curious, to ask questions, to search. You just give up. And generosity, helping other people? Forget it. You don't have any time at all. On the other side, our sense was good friction helps people to avoid choosing a myopic and overconfident version of themselves. That's why what good friction does is those obstacles, they lead to deliberation, thought, and as you said, they help you make wiser decisions. Okay, so you're talking about a lot of negative friction examples. What is the most common type of unnecessary friction in the workplace or in business that you found in your research? The most common type to me, it's essentially when organizations just add too much stuff on top of people. We call this addition sickness. Just to focus on one thing that's specifically driving me nuts lately is whether it's Stanford University or Google slash Alphabet. One of the reasons that they have friction problems, too many procedures, too much slowness, too much confusion, handoff problems, is that people at these organizations and thousands of others is the more people report to them, the more their bosses get paid. So the fiefdom spread and that ends up being people who add complexity to justify their existence. And I'm not even necessarily blaming them. It's just a human tendency that organizations and people have to resist. You know, if we think of obstacles that infuriate people, they range on the one hand from an endless series of back to back meetings. It could be long chains of approval. It could be delay in decision making. We opened the book, of course, with a classic email sent to all of us at Stanford by our own vice provost. How many words did it have? 1266 words, 7400 word attachment, requesting all 2000 faculty members to devote a Saturday to brainstorming about a new sustainability school. And I immediately started editing it and I got it down to 400 words in about three minutes because it was mostly her anticipating criticisms or repeating criticisms she's already had before. And if you just multiply, if it had been 400 words rather than 1266 words, just think about how much time she would have saved the people in her institution."
input_ids = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")["input_ids"]
summary_ids = model.generate(input_ids)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
  

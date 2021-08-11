from datasets import load_dataset, load_metric # 加载数据集和度量方法
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification # 从ckpt加载tokenizer和模型
from transformers import TrainingArguments, Trainer # 训练
import numpy as np
import torch

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# 准备传入trainer的各项参数
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2) # 这里之前还多了一个num_labels参数
training_args = TrainingArguments(
    "test-trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch" # 每个epoch都会测试一下
)

# 准备evaluate的一些过程
metric = load_metric("glue", "mrpc")
def compute_metrics(eval_preds):
    ''' 应该是对一个batch进行evaluate '''
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 构建一个trainer对象
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train() # 开始进行训练

# 训练后的预测(训练集，其它同理)
predictions = trainer.predict(tokenized_datasets["train"])
metric = load_metric("glue", "mrpc")
preds = np.argmax(predictions.predictions, axis=-1)
print("train set上的准确率和F1值是：", metric.compute(predictions=preds, references=predictions.label_ids))
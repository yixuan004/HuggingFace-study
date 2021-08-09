[toc]
# 2.6 Putting it all together
部分相关代码见：
> HuggingFaceStartCourses/2_Using Hugging Face Transformers/2_6_Putting_it_all_together.ipynb

## 2.6.1 Overview
在之前的几个章节中，我们一直在尽最大努力手工完成大部分工作，我们探讨了tokenizer是如何工作的，并研究了tokenize，转化为input_ids，padding，truncation，和attention_mask

但是，正如我们在第2节中看到的，Huggingface Transformers API可以通过一个高级函数为我们处理所有这些，我们将在这里深入讨论。当您直接在句子上调用tokenizer的时候，您将返回准备送入到模型的输入：
```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```
在这里，**model_inputs**变量包含使得模型能够正常运行的一切。对于DistillBERT，它包括input_ids和attention_mask。接受额外输入的其他模型也会有**tokenizer**对象的输出。

正如我们将在下面的一些示例中看到的，这种方法非常强大。首先，它可以tokenize单个序列：
```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
```

这个也可以同时处理多个同时的句子，并且不用换API
```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

model_inputs = tokenizer(sequence)
```

可以使用这几种参数进行padding
```python
# 会填充sequence到最长句子的那个长度，估计是默认的
model_inputs = tokenizer(sequences, padding="longest")

# 会填充sequence到最大长度max_length
# 512 for BERT or DistilBERT
model_inputs = tokenizer(sequences, padding="max_length")

# 会填充到一个指定的max_length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

tokenizer也可以截断序列：
```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# 会截断超过模型最大长度的序列
# 512 for BERT or DistilBERT
model_inputs = tokenizer(sequences, truncation=True)

# 会截断超过自己指定长度的序列
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

**tokenizer**对象可以处理特定框架张量的转换，然后可以直接发送到模型。例如，在下面的代码示例中，我们提示标记器从不同的框架返回张量 —— “pt”返回PyTorch张量，“tf”返回TensorFlow张量，“np”返回NumPy数组。
```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# return TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## 2.6.2 Special tokens

这里的special tokens可以指的是[CLS] [SEP]

如果我们看一下标记器返回的input_ids，我们会发现它们与之前的略有不同（看到多了一个101，102）：

```python
from transformers import AutoTokenizer

sequence = "I've been waiting for a HuggingFace course my whole life."

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# case1
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
>>> [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]

# case2
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
>>> [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```

在开始时添加了一个token ID，在结束时添加了一个token ID。让我们解码上边两个ID序列，看看这是怎么回事：
```python
from transformers import AutoTokenizer

sequence = "I've been waiting for a HuggingFace course my whole life."

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# case1
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
print(tokenizer.decode(model_inputs["input_ids"]))
>>> [CLS] i've been waiting for a huggingface course my whole life. [SEP]

# case2
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
print(tokenizer.decode(ids))
>>> i've been waiting for a huggingface course my whole life.
```

tokenizer在开头添加了特殊单词[CLS]，在结尾添加了特殊单词[SEP]，这是因为模型是用这些数据预训练的，所以为了得到相同的推断结果，我们还需要添加他们。请注意，有些模型不添加特殊单词，或添加不同的单词。模型也可以仅在开头或结尾添加这些特殊单词。在任何情况下，tokenizer都知道哪些需要tokenize，并将为您处理这些标记


## 2.6.3 Wrapping up: From tokenizer to model

现在我们已经了解了**tokenizer**对象应用于文本时使用的所有单独步骤，让我们最后一次性了解它如何处理多个序列（padding），非常长的序列（truncation），以及多种类型的tensor及主要API：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```
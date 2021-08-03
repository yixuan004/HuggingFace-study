# Quick Tour
Hugging Face 快速上手

代码见：
> QuickTour/Hugging_Face_Quick_Tour.ipynb

我们为快速使用模型提供了 pipeline （流水线）API。流水线聚合了预训练模型和对应的文本预处理。许多的 NLP 任务都有开箱即用的预训练流水线。

## 1. 一个快速使用pipeline去判断正负面情绪的例子

正向：
第二行代码下载并缓存了流水线使用的预训练模型，而第三行代码则在给定的文本上进行了评估。这里的答案“正面” (positive) 具有 99 的置信度。
```python
from transformers import pipeline

# 使用情绪分析流水线
classifier = pipeline('sentiment-analysis')
classifier('We are very happy to introduce pipeline to the transformers repository.')
```
output：
> [{'label': 'POSITIVE', 'score': 0.9996980428695679}]


负向：
```python
from transformers import pipeline

# 使用情绪分析流水线
classifier = pipeline('sentiment-analysis')
classifier('damn, how the weather is today!')
```
output：
> [{'label': 'NEGATIVE', 'score': 0.9843284487724304}]

中文尝试：
```python
from transformers import pipeline

# 使用情绪分析流水线
classifier = pipeline('sentiment-analysis')
a = classifier('今天的天气真好啊！')
print(a)
b = classifier('今天的天气不太好，郁闷啊！')
print(b)
```
output:
> [{'label': 'POSITIVE', 'score': 0.7827630639076233}]
[{'label': 'NEGATIVE', 'score': 0.7204760909080505}]


## 2. 从给定文本中抽取问题答案

<font color='red'>
首次接触到这种答案抽取问答的任务，对于问答、对话这种未来的潜在方向也要多学习&nbsp
</font>
<br>
<br>
除了给出答案，预训练模型还给出了对应的置信度分数、答案在词符化 (tokenized) 后的文本中开始和结束的位置。

英文
```python
from transformers import pipeline

# 使用问答流水线
question_answerer = pipeline('question-answering')
question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
    })
```
output：
> {'answer': 'huggingface/transformers',
 'end': 58,
 'score': 0.30970120429992676,
 'start': 34}


中文
```python
from transformers import pipeline

# 使用问答流水线
question_answerer = pipeline('question-answering')
question_answerer({
    'question': '勒克莱尔在哪里退赛了',
    'context': '勒克莱尔在匈牙利大奖赛中退赛'
    })
```
output：
> {'answer': '中退赛', 'end': 14, 'score': 0.009904472157359123, 'start': 11}

<font color='red'>
从这个例子中可以看出来，pipeline对于中文的问答支持还不是很好
</font>

## 3. 更多流水线API支持的任务

见：https://huggingface.co/transformers/task_summary.html


## 4. 在你的任务上下载和使用任意预训练模型

### 4.1 Pytorch任务中使用任意预训练模型

要在各类NLP任务上下载和使用任意预训练模型也很简单，只需三行代码。这里是 PyTorch 版的示例：
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
# print("inputs：", inputs) 
# >>> {'input_ids': tensor([[ 101, 7592, 2088,  999,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}

outputs1 = model(**inputs)
# equals to
outputs2 = model(inputs['input_ids']) # 为什么token_type_ids和attention_mask没有用到，可能还需要根据之后的教程进一步学习了

print(outputs1['last_hidden_state'] == outputs2['last_hidden_state'])
>>> tensor([[[True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True]]])

print("outputs1['last_hidden_state'].shape:", outputs1['last_hidden_state'].shape)
print("outputs1['pooler_output'].shape:", outputs1['pooler_output'].shape)

>>> outputs1['last_hidden_state'].shape: torch.Size([1, 5, 768])
>>> outputs1['pooler_output'].shape: torch.Size([1, 768])

# BaseModelOutputWithPoolingAndCrossAttentions对象
# print(outputs) 
```

tokenizer（词符化器）为所有的预训练模型提供了预处理，并可以直接对单个字符串进行调用，或对列表（list）进行调用。它会输出一个你可以在下游代码里使用，或直接通过 ** 解包表达式传给模型的词典（dict）

<font color='red'>
这里补充一个之前不太常用的内容，python字典前加*，**的作用
</font>
```python
def add(a, b):
    return a + b

data = [4, 3]
print(add(*data))
# equals to print add(4, 3)

data = {'a': 4, 'b': 3}
print(add(*data))
# equals to print add('a', 'b')
print(add(**data))
# equals to print add(4, 3)
```

模型本身是一个常规的 Pytorch nn.Module（或 TensorFlow tf.keras.Model），可以常规方式使用。 [这个教程](https://huggingface.co/transformers/training.html)解释了如何将这样的模型整合到经典的 PyTorch 或 TensorFlow 训练循环中，或是如何使用我们的 Trainer 训练器）API 来在一个新的数据集上快速微调。

<font color="red">
上边这个教程的地方，将在UsingTransformers/目录下进行更为详细的学习
</font>


### 4.2 TensorFlow任务中使用任意预训练模型
待补充
# 1.8 Bias and limitations
偏见和限制

如果您打算在生产中使用经过预训练的模型或经过微调的版本，请注意，虽然这些模型是强大的工具，但它们也有局限性。其中最大的一个问题是，为了对大量数据进行预培训，研究人员通常会搜集所有他们能找到的内容，从互联网上可用的内容中选择最好的和最坏的。

为了快速演示，让我们回到使用BERT模型的**mask-filling** pipeline的示例：

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

>>> ['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

>>> ['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

当要求填写这两句话中缺少的单词时，模型只给出一个无性别的答案（服务员/女服务员）。其他职业通常与某一特定性别相关——是的，妓女最终进入了模型中与“女人”和“工作”相关的前五位。尽管伯特是一个罕见的Transformer模型，但它不是通过从互联网上搜集数据建立的，而是使用明显中立的数据（它是在英语维基百科和图书语料库数据集上训练的）。

因此，当您使用这些工具时，您需要记住，您使用的原始模型很容易生成性别歧视、种族主义或恐同内容。根据数据微调模型不会使这种固有偏差消失。
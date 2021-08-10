# 3.1 Introduction
在[第2节Using Hugging Face Transformers](https://huggingface.co/course/chapter2)中，我们探讨了如何使用tokenizer和预训练模型**进行预测**。但是，如果您想为自己的数据集微调预训练模型，该怎么办？这就是本章的主题，将学习：

* 如何从中心准备大型数据集
* 如何使用高层级的**Trainer**API来fine-tune模型
* 如何自定义一个training loop
* 如何利用HuggingFace Accelerate库在任何分布式设置上轻松运行自定义训练循环
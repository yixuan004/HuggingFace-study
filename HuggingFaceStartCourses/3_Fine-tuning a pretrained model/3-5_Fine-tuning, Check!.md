# 3.5 Fine-tuning, Check!

真有趣！在前两章中，您学习了model和tokenizer，现在您知道如何根据自己的数据对它们进行fine-tune。总而言之，在本章中，可以做到：

* 了解了[hub](https://huggingface.co/datasets)中的数据集
* 学习了如何加载和预处理数据集，包括使用dynamic padding和collators
* 实现了自己模型的fine-tune和evaluate
* 实现了一个low-level的训练循环
* 使用HuggingFace Accelerate来轻松调整训练循环，适用于多个GPU或者TPU
# 2.1 Introduction

在第1章节中，我们使用high-level的pipeline API来为不同的任务使用Transformer模型。尽管此API功能强大且方便，但了解它在后台的工作方式非常重要，这样我们就可以灵活地解决其他问题。

在本章节中，将学习到：

* 怎么使用tokenizers和models来仿制一个pipeline API的行为
* 怎么加载和保存models和tokenizers
* 不同的token化方法，例如word-based, character-based, subword-based
* 如何处理多个不同长度的句子

## 2.1.1 Introduction

就像章节1中所见到的，Transformer的模型通常是非常巨大的。由于有数百万到数百亿个参数，训练和部署这些模型是一项复杂的任务。此外，由于新模型几乎每天都在发布，而且每种模型都有自己的实现，因此尝试它们绝非易事。

Hugging Face Transformers库（library）的建立就是为了解决这个问题。它的目标是提供一个API，通过它可以加载、训练和保存任何Transformer模型。这个library的主要特点是：

* Ease of use 便于使用：下载、加载和使用最先进的NLP模型进行推理只需两行代码即可完成。
* Flexibility 灵活性：在其核心，所有模型都是简单的PyTorch nn.Module或TensorFlow tf.keras.Model类，并且可以像各自机器学习（ML）框架中的任何其他模型一样进行处理
* Simplicity 简单化：整个library几乎没有任何抽象化（abstractions）。“一个文件中的所有内容”是一个核心概念：模型的前向传递完全定义在一个文件中，因此代码本身是可以理解的，并且是可以破解的。

最后一个特性使Hugging Face Transformers与其他ML库完全不同。模型不是建立在跨文件共享的模块上的；相反，每个模型都有自己的层。除了使模型更容易接近和理解之外，这还允许您在不影响其他模型的情况下轻松地在一个模型上进行实验。

<font color="red">
自：如果不抽象化的话，理解起来可能会好理解很多，因为就不涉及嵌套在一起的问题了。但是在实现上可能会带来很大的工程，作者可能维护了一种很好很方便的接口？
</font>

本章将从一个端到端的示例开始，在该示例中，我们一起使用一个model和一个tokenizer来复制第1章中介绍的**pipelien** API。接下来，我们将讨论模型API：我们将深度研究模型和配置类，并向您展示如何加载模型以及它如何处理数字输入以输出预测。

然后我们来看看tokenizer的API，他是pipeline的另一个主要组件。tokenizer负责第一个和最后一个处理步骤，处理从文本到神经网络数值化的转换，以及在需要时转换回文本。最后，我们将向您展示如何处理在一个准备好的批处理过程中，通过一个模型发送多个句子，然后通过更详细的了解更high level的**tokenizer**来总结这一切。
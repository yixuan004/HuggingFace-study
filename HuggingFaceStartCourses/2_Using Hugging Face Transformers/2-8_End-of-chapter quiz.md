[toc]
# 2.8 End-of-chapter quiz

## 2.8.1 自选择+笔记

1：注意，首先拿到的是一些prediction，之后可能要再通过一次tokenizer才能转化为text
![](./MarkdownPictures/2021-08-09-13-56-53.png)

2：想一想那个[1, 14, 768]的例子，1是batch_size，14是长度，768是hidden_size
![](./MarkdownPictures/2021-08-09-13-58-18.png)

3：subword级别是现在最常用的，能综合一些优缺点，主要是弥补word级别的OOV等不足
![](./MarkdownPictures/2021-08-09-13-59-49.png)

4：model head感觉可以理解为经过transformer后的，针对指定任务的一个head的感觉
![](./MarkdownPictures/2021-08-09-14-04-18.png)

5：from transformers import AutoModel
![](./MarkdownPictures/2021-08-09-14-05-12.png)

6：这里truncation也是一种方法，自己漏选了
![](./MarkdownPictures/2021-08-09-14-06-07.png)

7：
![](./MarkdownPictures/2021-08-09-14-07-26.png)

8：核心是直接调用tokenizer()
![](./MarkdownPictures/2021-08-09-14-08-48.png)

9：
![](./MarkdownPictures/2021-08-09-14-09-37.png)

10：对这个例子来说，输入是一句而不是一个batch
![](./MarkdownPictures/2021-08-09-14-12-54.png)

## 2.8.2 全部答案+说明截图

1：
![](./MarkdownPictures/2021-08-09-14-13-43.png)

2：
![](./MarkdownPictures/2021-08-09-14-13-58.png)

3：
![](./MarkdownPictures/2021-08-09-14-14-38.png)

4：
![](./MarkdownPictures/2021-08-09-14-14-53.png)

5：
![](./MarkdownPictures/2021-08-09-14-15-09.png)

6：
![](./MarkdownPictures/2021-08-09-14-15-35.png)

7：
![](./MarkdownPictures/2021-08-09-14-16-14.png)

8：
![](./MarkdownPictures/2021-08-09-14-16-33.png)

9：
![](./MarkdownPictures/2021-08-09-14-17-11.png)

10：
![](./MarkdownPictures/2021-08-09-14-17-33.png)


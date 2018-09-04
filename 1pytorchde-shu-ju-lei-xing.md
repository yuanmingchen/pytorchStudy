# 一、pytorch基础知识

### 1、pytorch的数据类型

pytorch的数据类型是`torch.Tensor`，\(`torch.Tensor`是默认的tensor类型`torch.FlaotTensor`的简称。\)这是一种包含单一数据类型元素的多维矩阵。它的中文名字叫做张量，什么是张量呢？

张量可以作为“表格”理解，

* 0维张量是一个数，又称标量；比如数字5
* 1维张量是一个只有一行的表格，又称向量；当然向量也可以只有一个数，比如向量\[1,2,0,3\]
* 二维张量是多行多列的表格，又称矩阵；矩阵也可以只有一行,比如矩阵
* \[\[1,2,3\],
* \[3,4,2\]\]
* 三维张量可以看做多个表格，也可以只有一个表格
* ………

所以单单一个数可能是任意维的张量，所以张量必须包含维度信息。比如：

```py
#1:标量，0维张量
#[1]：向量，1维张量
#[[1]]:矩阵，2维张量
#[[[1]]]:三维张量
```

### 2、创建一个张量`torch.Tensor`

如何创建一个`torch.Tensor`？一个张量tensor可以从Python的`list`或序列构建：

```
>>> torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
1 2 3
4 5 6
[torch.FloatTensor of size 2x3]
```

一个空张量tensor可以通过规定其大小来构建：

```
>>> torch.IntTensor(2, 4).zero_()
0 0 0 0
0 0 0 0
[torch.IntTensor of size 2x4]
```

可以用python的索引和切片来获取和修改一个张量tensor中的内容：

```
>>> x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]) #一般使用FloatTensor
>>> print(x[1][2])
6.0
>>> x[0][1] = 8
>>> print(x)
 1 8 3
 4 5 6
[torch.FloatTensor of size 2x3]
```

每一个张量tensor都有一个相应的`torch.Storage`用来保存其数据。类tensor提供了一个存储的多维的、横向视图，并且定义了在数值运算。

根据可选择的大小和数据新建一个tensor。 如果没有提供参数，将会返回一个空的零维张量。如果提供了`numpy.ndarray`,`torch.Tensor`或`torch.Storage`，将会返回一个有同样参数的tensor.如果提供了python序列，将会从序列的副本创建一个tensor。


# 二、常用的数据处理函数

## 1.`torch.from_numpy(ndarray) → Tensor`

#### （1）介绍：

数据类型转化函数，将`numpy.ndarray`转换为pytorch的`Tensor`。 返回的张量tensor和numpy的ndarray**共享同一内存空间**。修改一个会导致另外一个也被修改。**返回的张量不能改变大小**。

#### （2）参数和返回类型

参数是`numpy.ndarray`类型，返回值是pytroch的张量类型。

```
torch.from_numpy(ndarray) → Tensor
```

#### （3）举例：

```py
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
torch.LongTensor([1, 2, 3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])
```

## 2.一些常用的运算函数

#### （1）`torch.abs(input, out=None) → Tensor`

##### a.介绍：

计算输入张量的每个元素绝对值

##### b.举例：

```
>>> torch.abs(torch.FloatTensor([-1, -2, 3]))
FloatTensor([1, 2, 3])
```

#### （2）`torch.add(input, value, out=None)`

##### a.介绍

对输入张量input逐元素加上标量值value，并返回结果到一个新的张量out，即 $$out=tensor+value$$。如果输入input是FloatTensor or DoubleTensor类型，则value 必须为实数，否则须为整数。【译注：似乎并非如此，无关输入类型，value取整数、实数皆可。】

##### b.参数及返回值：

* input \(Tensor\) – 输入张量，**必须参数**
* value \(Number\) – 添加到输入每个元素的数，**必须参数**
* out \(Tensor, optional\) – 结果张量，**可选参数**
* **返回值：结果张量**

```
>>> a = torch.randn(4)
>>> a

 0.4050
-1.2227
 1.8688
-0.4185
[torch.FloatTensor of size 4]

>>> torch.add(a, 20)

 20.4050
 18.7773
 21.8688
 19.5815
[torch.FloatTensor of size 4]
```

#### （3）`torch.add(input, value=1, other, out=None)`

`other` 张量的每个元素乘以一个标量值`value`，并加到`input` 张量上。返回结果到输出张量`out`。即，$$out=input+(other∗value)$$

两个张量`input`and`other`的尺寸不需要匹配，但**元素总数**必须一样。

**注意**:当两个张量形状不匹配时，输入张量的形状会作为输出张量的尺寸。

\(2\)中的add相当于other默认是一个全为1的张量

## 3.`torch.linspace(start, end, steps=100, out=None) → Tensor`

#### （1）介绍

返回一个1维张量，包含在区间`start`和`end`上均匀间隔的`steps`个点。 输出1维张量的长度为`steps`。

#### （2）参数

* start \(float\) – 序列的起始点
* end \(float\) – 序列的最终值
* steps \(int\) – 在start 和 end间生成的样本数
* out \(Tensor, optional\) – 结果张量

#### （3）举例

```py
>>> torch.linspace(3, 10, steps=5)

  3.0000
  4.7500
  6.5000
  8.2500
 10.0000
[torch.FloatTensor of size 5]

>>> torch.linspace(-10, 10, steps=5)

-10
 -5
  0
  5
 10
[torch.FloatTensor of size 5]

>>> torch.linspace(start=-10, end=10, steps=5)

-10
 -5
  0
  5
 10
[torch.FloatTensor of size 5]
```

## 4.`torch.nn.functional`

这个模块内含有许多常用的函数，包括激活函数、损失函数等多个函数。

导入模块：

```
import torch.nn.functional as F
```

举例：

以relu函数为例：

```py
torch.nn.functional.relu(input, inplace=False) → Tensor
```

```py
import torch.nn.functional as F
x = F.relu(x)  #x经过激活函数relu处理后赋值给x
```

还有一些常用的函数

```py
#线性函数
torch.nn.functional.linear(input, weight, bias=None)

#损失函数
torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)

#常用的激活函数
torch.nn.functional.relu(input, inplace=False)
torch.nn.functional.tanh(input)
torch.nn.functional.sigmoid(input)
torch.nn.functional.softplus(input, beta=1, threshold=20)
torch.nn.functional.softmax(input)
```

更多函数请参考：

[https://pytorch-cn.readthedocs.io/zh/latest/package\_references/functional/](https://pytorch-cn.readthedocs.io/zh/latest/package_references/functional/)




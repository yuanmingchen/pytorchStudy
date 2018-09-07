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

这里顺便介绍一个与之很类似的函数`torch.arange(start, end, step=1, out=None) → Tensor`

它的参数与`linspace`是乍一看的，但其实第三个参数不一样，`linspace`的第三个参数是steps，代表划分的个数，而`arange`的第三个参数是step，代表步长，即从`start`到`end`每隔step取一个数，步长默认为1。

另外有一点不一样的是，`linspace`是左闭右闭的，而`arange`是左闭右开的，即取的点包括`start`不包括`end`。

```
>>> torch.arange(1, 4)  #省略步长的情况，左闭右开

 1
 2
 3
[torch.FloatTensor of size 3]

>>> torch.arange(1, 2.5, 0.5)

 1.0000
 1.5000
 2.0000
[torch.FloatTensor of size 3]
```

还有一个与torch.arange非常相似的函数是`torch.range(start, end, step=1, out=None) → Tensor`

唯一的不同之处在于它返回的数据是左闭右闭的。

```py
>>> torch.range(1, 4)

 1
 2
 3
 4
[torch.FloatTensor of size 4]

>>> torch.range(1, 4, 0.5)

 1.0000
 1.5000
 2.0000
 2.5000
 3.0000
 3.5000
 4.0000
[torch.FloatTensor of size 7]
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

## 5.unsqueeze函数

```
torch.unsqueeze(input, dim, out=None)
```

#### （1）介绍

返回一个新的张量，对输入的制定位置插入维度 1

**注意：** 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。

如果`dim`为负，则将会被转化\( dim+input.dim\(\)+1\)

#### （2）参数：

* tensor \(Tensor\) – 输入张量
* dim \(int\) – 插入维度的索引
* out \(Tensor, optional\) – 结果张量

#### （3）举例：

```
>>> x = torch.Tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
 1  2  3  4
[torch.FloatTensor of size 1x4]
>>> torch.unsqueeze(x, 1)
 1
 2
 3
 4
[torch.FloatTensor of size 4x1]
```

#### （4）个人理解

上面是官方文档，感觉还是不好理解，下面是我自己的理解：

```py
unsqueeze函数是解压函数，所谓解压函数就是把输入的张量在指定的位置上插入1个维度，还是很难理解吧

以二维张量[[0,1,2],[3,4,5]]为例，由于0维的值一般都为1，我们忽略它，所它是2*3的一个张量
它可以选择分解的位置为0,1,2这三个位置：2之前，2和3之间，3之后这三个位置，分别记作0,1,2位置
如果dim为负数，则将会被转化为(dim+input.dim()+1)

何为插入一个维度，我们的x是2*3（从高维到低维）的二维向量，在2号位置插入1个维度意味着让它变成2*3*1
下面看详细举例
>>> x = torch.Tensor([[0,1,2],[3,4,5]])
>>> torch.unsqueeze(x, 0)
tensor([[[0., 1., 2.],
         [3., 4., 5.]]])    #变成1*2*3
>>> torch.unsqueeze(x, 1)
tensor([[[0., 1., 2.]],

        [[3., 4., 5.]]])   #变成2*1*3
>>> torch.unsqueeze(x, 2)
tensor([[[0.],
         [1.],
         [2.]],

        [[3.],
         [4.],
         [5.]]])         #变成2*3*1
我们发现：
dim = 0：变成1*2*3，要求第三维为1，其余维不变，直接把矩阵变成只包含一个矩阵的三维张量即可
dim = 1：变成2*1*3，第一维不变，第二维变成1，即标量还是标量，
         向量变成单行矩阵，两个矩阵自然变成了一个三维张量
dim = 2：变成2*3*1，一二三维都要改变，0维的标量变只有一个数的向量，
         而向量自然而然的就变成了矩阵，矩阵自然而然的变成了三维张量
注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
```

## 6.squeeze函数

#### （1）介绍：

squeeze函数是压缩函数，把输入张量中值为1的维度删除，这个较为简单。

官方解释：

将输入张量形状中的`1`去除并返回。 如果输入是形如$$((A \times 1\times B \times 1 \times C \times 1 \times D) )$$，那么输出形状就为： $$((A \times B \times C \times D) )$$

当给定`dim`时，那么挤压操作只在给定维度上。例如，输入形状为: $$((A \times 1 \times B) )$$,`squeeze(input, 0)`将会保持张量不变，只有用`squeeze(input, 1)`，形状会变成$$( (A \times B ))$$。

**注意：** 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。

#### （2）参数:

* input \(Tensor\) – 输入张量
* dim \(int, optional\) – 如果给定，则`input`只会在给定维度挤压
* out \(Tensor, optional\) – 输出张量

#### （3）举例：

```py
#官方举例
>>> x = torch.zeros(2,1,2,1,2)
>>> x.size()
(2L, 1L, 2L, 1L, 2L)
>>> y = torch.squeeze(x)
>>> y.size()
(2L, 2L, 2L)
>>> y = torch.squeeze(x, 0)
>>> y.size()
(2L, 1L, 2L, 1L, 2L)
>>> y = torch.squeeze(x, 1)
>>> y.size()
(2L, 2L, 1L, 2L)

>>> x = torch.zeros(2,1,2,1,2)  zeros函数返回一个2*1*2*1*2的全是0的张量
tensor([[[[[0., 0.]],

          [[0., 0.]]]],



        [[[[0., 0.]],

          [[0., 0.]]]]])
>>> torch.squeeze(x)   #压缩后变成2*2*2的张量，每个向量2个数，每个矩阵2行，每个三维张量含2个矩阵，5维张量变成3维
tensor([[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]])

#举例2：2维变1维，把一个两行的矩阵压缩成一个向量
>>> x = torch.zeros(2,1)
tensor([[0.],
        [0.]])
>>> torch.squeeze(x)
tensor([0., 0.])
```

## 7.`torch.ones(*sizes, out=None) → Tensor`

#### （1）介绍：

返回一个全为1 的张量，形状由可变参数`sizes`定义。

#### （2）参数

* sizes \(int...\) – 整数序列，定义了输出形状
* out \(Tensor, optional\) – 结果张量 

#### （3）举例

```py
>>> torch.ones(2, 3)

 1  1  1
 1  1  1
[torch.FloatTensor of size 2x3]

>>> torch.ones(5)
 1
 1
 1
 1
 1
[torch.FloatTensor of size 5]
```

与之类似的产生指定数据的张量的函数还有：

**torch.zeros**，唯一不同之处在于返回的张量数据都为0。

**torch.randn：**返回包含了从区间\[0,1\)的均匀分布中抽取的一组随机数，形状由可变参数`sizes`定义。

## 8、`torch.normal(means, std, out=None)`

#### （1）介绍

返回一个张量，包含从给定参数`means`,`std`的离散正态分布中抽取随机数。 均值`means`是一个张量，包含每个输出元素相关的正态分布的均值。`std`是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同。

当means与std的形状不一样时，返回的结果与means的形状相同。

#### （2）参数

* means \(Tensor\) – 均值，默认值为0.0
* std \(Tensor\) – 标准差，默认值为1.0
* out \(Tensor\) – 可选的输出张量

#### （3）举例

```py
#means和std的形状一样，一一对应，返回的结果形状是（1*10）
torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))

 1.5104
 1.6955
 2.4895
 4.9185
 4.9895
 6.9155
 7.3683
 8.1836
 8.7164
 9.8916
[torch.FloatTensor of size 10]

#means不是张量，而是一个数，那么所有数据都公用同一个均值0.5，返回的数据个数与std的形状相同
>>> torch.normal(mean=0.5, std=torch.arange(1, 6))
  0.5723
  0.0871
 -0.3783
 -2.5689
 10.7893
[torch.FloatTensor of size 5]


#std被省略，使用默认值std = 1.0，返回的结果与means形状相同
>>> torch.normal(means=torch.arange(1, 6))

 1.1681
 2.8884
 3.7718
 2.5616
 4.2500
[torch.FloatTensor of size 5]

#当means与std的形状不一样时，返回的结果与means的形状相同
import torch
x = torch.ones(2,4)
y = torch.ones(1,8)
out = torch.normal(x,y)
print(out)
#输出：
tensor([[ 0.4471,  2.3478,  0.7964, -1.3335],
        [ 2.6741,  2.1961, -0.3855, -0.4517]])
```

#### （4）个人理解

其实对于means和std的形状并没有太大的限制，限制的是他们含有的数据是否可以一一对应。

means:2\*3    std:1\*6      OK  注意，此时返回的结果的形状与means相同，也就是2\*3。

means:任意形状的张量  std：任意常数或省略  OK

means:任意常数或省略  std：任意形状的张量  OK

只有means和std都是张量，并且含有的数据个数还不一样才不可以。

## 9、`torch.cat`

#### （1）介绍

`torch.cat(inputs, dimension=0) → Tensor`

cat方法在**给定维度上**对输入的张量序列`seq`进行**连接操作**。`torch.cat()`可以看做`torch.split()`和

`torch.chunk()`的反操作。`cat()`函数可以通过下面例子更好的理解。

#### （2）参数

* inputs \(sequence of Tensors\) – 可以是任意相同Tensor 类型的python 序列
* dimension \(int, optional\) – 沿着此维连接张量序列。

#### （3）举例

```py
>>> x = torch.randn(2, 3)
>>> x

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x3]

>>> torch.cat((x, x, x), 0)

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 6x3]

>>> torch.cat((x, x, x), 1)

 0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x9]
```

#### （4）个人理解

这里需要注意的是dimension这个参数，我认为这个参数指的是维度的编号，对于示例中的x而言，x是一个2\*3的张量，0代表在‘2’这个维度进行连接，1代表在‘3’这个维度进行连接。




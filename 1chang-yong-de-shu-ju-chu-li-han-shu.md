### 二、常用的数据处理函数

## 1.`from_numpy(numpy.ndarray)`

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

## 2.




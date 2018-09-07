# 五、分类

## 1、介绍

这里介绍的是最简单的二分类问题，我们只有两个类，因此非常简单，可以把问题看做对于一个输出，模型输出0和1的概率，0和1分别代表两个类的编号。

## 2、神经网络实现

```py
#Method 1
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x
```

这个神经网络和回归的神经网络基本上是一样的，也是只有一个输入层和输出层，使用了relu函数作为激活函数，没有什么新鲜的东西。那么我们在这里给出实现该神经网络的另外一种写法，这种方式更加简便：

```py
#Method2  Sequential是一个类，一个时序容器，可以接受任意个参数，
# 每个参数是一个处理模块（一层或一个激活函数），各模块会以他们被传入的书序添加到容器中
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),  #创建一个神经层（输入层）
    torch.nn.ReLU(), #增加一个激励函数
    torch.nn.Linear(10,2),  #输出层
)
```

第二种方法使用了`torch.nn.Sequential`这个类，这一个时序容器。

在我们上面的神经网络中，首先会对传入的数据进行线性变换，输出10个数据，然后把输出用relu函数激活，然后再次进行线性变换，输出2个数据。

这里需要解释一下为什么是两个数据，因为分类问题实际上的输出，实际上输出是对各个分类的预测值，可以看做是数据被判定维各个分类的概率（即使输出不在0-1范围之内，也可以使用softmax函数将其变成类似于概率的数据），比如我们现在有三个类A、B、C，对于输入数据x，我们的模型输出可能是

（11,43,5），于是模型就将其判定为可能性最大的B类。

我们的问题是一个简单的二分类问题，所以最终的输出预测结果就是二维的。

`torch.nn.Sequential`这个类，是一个时序容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict有序字典。

为了更容易的理解如何使用Sequential, 下面给出了一个例子:

```py
# Example of using Sequential
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

## 3、神经网络的使用

首先我们来看一下输入的数据，这里造了一些假数据：

```py
#定义了一个100*2的张量，返回一个全为1 的张量，形状由可变参数sizes = 100*2定义。
n_data = torch.ones(100,2) 

#返回从单独的正态分布中提取的随机数的张量，第一个参数是均值，第二个参数是标准差。2-1到2+1
x0 = torch.normal(2*n_data,1)
# 定义了一个100维的向量，返回一个全为标量 0 的张量，形状由可变参数sizes 定义。
y0 = torch.zeros(100) 

#返回从单独的正态分布中提取的随机数的张量，第一个参数是均值，第二个参数是标准差。-2-1到-2+1
x1 = torch.normal(-2*n_data,1)
# 定义了一个100*2的张量，返回一个全为1 的张量，形状由可变参数sizes = 100*2定义。
y1 = torch.ones(100)

# 在给定维度上对输入的张量序列seq 进行连接操作。返回一个200*2的张量
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  
# shape (200,)返回一个200的张量，钱100个数是0，后100个数是1
y = torch.cat((y0, y1), ).type(torch.LongTensor)
```

解释一下：

`torch.ones(100,2)`：这个函数返回了一个100\*2的张量，张量里面的数据都为1

`torch.normal(2*n_data,1)`：返回从单独的正态分布中提取的随机数的张量，第一个参数是均值，第二个参数是标准差。

`2*n_data`的意思是把`n_data`中的每个数都乘以二，`n_data`的形状并不会发生改变，还是一个100\*2的张量，返回的结果也是100\*2的张量。

`torch.zeros(100)` ：返回一个包含100个数的张量，张量里面的数据都为0。

x = torch.cat\(\(x0, x1\), 0\).type\(torch.FloatTensor\)  






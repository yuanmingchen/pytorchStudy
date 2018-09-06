# 五、分类

## 1、介绍

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
    torch.nn.Linear(2,10),  #创建一个神经层（隐藏层）
    torch.nn.ReLU(), #增加一个激励函数
    torch.nn.Linear(10,2),  #输出层
)
```

第二种方法使用了

## 3、神经网络的使用






# 四、回归拟合函数

## 1、介绍

最简单的神经网络使用，给我们一组数据，我们进行拟合。

## 2、神经网络定义

```py
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        #这是一层， 第一个参数是输入个数，第二个是输出个数
        self.hidden = torch.nn.Linear(n_features,n_hidden)
         #这是另一层，接受hidden层的输入，给出输出结果
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self, x):
        # 首先让隐藏层处理输入,经过激活函数激活作为本层输出
        x = F.relu(self.hidden(x)) 
        #预测（输出）层接受隐藏层处理后的数据再次进行处理，然后给出预测结果
        #为了防止输出结果被截断，这里不做激活函数处理
        x = self.predict(x) 
        return x
```

这个类有两个函数，第一个是构造方法`__init__`函数，神经网络各层的定义是在\_\_init\_\_函数中定义的，需要给出各层的输入输出个数。

另一个函数就是forward函数，这是神经网络具体执行的函数，在这里调用我们在\_\_init\_\_中定义的神经层。x是神经网络的输入，我们利用各神经层和激活函数在这个方法中进行处理，最后给出一个输出结果。

这个神经网络十分简单，只有一个输入层和一个输出层，n\_features是输入数据的个数，n\_hidden是输入层的输出个数，同时也是它的下一层——输出层的输入个数，n\_output是输出层的输出个数。

## 3、神经网络的使用

神经网络的使用也很简单，只需要新建一个神经网络，然后使用合适的优化器，计算误差，然后误差逆向传播，不断调整即可。

```
net = Net(1,10,1)  #分别代表输入值个数、隐藏层神经元数和输出个数

optimizer = torch.optim.SGD(net.parameters(),lr = 0.5)#选择SGD优化器，lr = 0.5是学习率
loss_func = torch.nn.MSELoss()  # 均方差作为我们的损失函数

for i in range(100): #一共执行100次优化过程
    prediction = net(x)  #获取神经网络的输出，也就是预测结果

    loss = loss_func(prediction,y) #计算预测结果与真实值得误差

    optimizer.zero_grad()  #梯度清零，以便于下一次计算
    loss.backward()   #误差逆向传播
    optimizer.step()  #根据设置的学习率，由优化器进行参数优化
```




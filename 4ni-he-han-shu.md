# 四、拟合函数

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




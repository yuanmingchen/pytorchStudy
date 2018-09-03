# 1、第一章 cnn

首先看一个CNN的实现过程：

```py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #卷积层
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28) 1是维度，28*28是宽高
            nn.Conv2d(                      #过滤器
                in_channels=1,              # input height 输入通道数，图片的层数，黑白图片是1，RGB是3
                out_channels=16,            # n_filters 输出通道数
                kernel_size=5,              # filter size  过滤器的宽高都设为5
                stride=1,                   # filter movement/step   每隔多少像素调一下，即每次移动一个像素
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1，在数据周围加一圈为0的数据
            ),                              # output shape (16, 28, 28) 原图变成了(16, 28, 28)
            nn.ReLU(),                      # activation 激活函数
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)  筛选想要的部分，选择2*2之间最大值作为特征，也可以选择平均值，一般用最大值
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes，需要把输入的三维数据展平成一维，在forward中实现

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7) 数据展平过程
        output = self.out(x)
        return output, x    # return x for visualization
```

上面的卷积神经网络有两个卷积层（即self.conv1和self.conv2）和一个输出层





## 2、函数讲解

### nn.Sequential\(\* args\)： {#class-torchnnsequential-args}






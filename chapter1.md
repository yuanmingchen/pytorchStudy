### 

# 1、第一章 cnn

[https://blog.csdn.net/ice\_actor/article/details/78648780\#commentBox](https://blog.csdn.net/ice_actor/article/details/78648780#commentBox)

这篇博客写的非常清楚了，看完以后再看代码好多了。

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

上面的卷积神经网络有两个卷积层（即self.conv1和self.conv2）和一个输出层。

## 2、函数讲解

### （1）nn.Sequential\(\* args\)： {#class-torchnnsequential-args}

这是一个时序容器。`Modules`会以他们传入的顺序被添加到容器中。当然，也可以传入一个`OrderedDict`。

为了更容易的理解如何使用`Sequential`, 下面给出了一个例子:

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

总之就是当数据进入`Sequential`所定义的神经层时候，就会把数据交给`Sequential`的模块依次进行处理，最后进行输出

### （2）class torch.nn.Conv2d\(in\_channels, out\_channels, kernel\_size, stride=1, padding=0, dilation=1, groups=1, bias=True\)

shape:  
输入: $$ (N,C_{in},H_{in},W_{in}) $$  
输出: $$(N,C_{out},H_{out},W_{out})$$  
输入输出的计算方式：  
$$H_{out}=floor((H_{in}+2padding[0]-dilation[0](kernerl_{size}[0]-1)-1)/stride[0]+1)$$

$$W_{out}=floor((W_{in}+2padding[1]-dilation[1](kernerl_{size}[1]-1)-1)/stride[1]+1)$$

**说明**  
`bigotimes`: 表示二维的相关系数计算`stride`: 控制相关系数的计算步长  
`dilation`: 用于控制内核点之间的距离，详细描述在[这里](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)  
`groups`: 控制输入和输出之间的连接：`group=1`，输出是所有的输入的卷积；`group=2`，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。

参数`kernel_size`，`stride,padding`，`dilation`也可以是一个`int`的数据，此时卷积height和width值相同;也可以是一个`tuple`数组，`tuple`的第一维度表示height的数值，tuple的第二维度表示width的数值

weight\(tensor\) - 卷积的权重，大小是\(out\_channels, in\_channels,kernel\_size\)

bias\(tensor\) - 卷积的偏置系数，大小是（out\_channel）

参数：

* in\_channels\(int\) – 输入信号的通道（黑白图片信道数为1），**必须参数**
* out\_channels\(int\) – 卷积产生的通道（输出信道数）**必须参数**
* kerner\_size\(int or tuple\) - 卷积核的尺寸，扫描器的长宽 **必须参数**
* stride\(int or tuple, optional\) - 卷积步长，相邻两次扫描的间隔
* padding\(int or tuple, optional\) - 输入的每一条边补充0的层数，为输入加上一圈边框，为了扫描之后输出的宽高与原来一样，以stride=5为例，而图片也是7\*7的，那么输出的宽高就成了3\*3,宽高减少了\(kerner\_size-1\)个，由于每次步长为1，所以每多加一个像素就可以让长宽加一,所以此时padding = \(kerner\_size-1\)/2\(因为左右都要加边框，所以除以2\)
* dilation\(int or tuple, optional\) – 卷积核元素之间的间距
* groups\(int, optional\) – 从输入通道到输出通道的阻塞连接数
* bias\(bool, optional\) - 如果bias=True，添加偏置

### （3）class torch.nn.ReLU\(inplace=False\)

对输入运用修正线性单元函数$${ReLU}(x)= max(0, x)$$，shape：

* 输入：$$(N, )$$，代表任意数目附加维度
* 输出：$$(N, *)$$，与输入拥有同样的shape属性

### （4）class torch.nn.MaxPool2d\(kernel\_size, stride=None, padding=0, dilation=1, return\_indices=False, ceil\_mode=False\)

最大池化函数，实际上也可以看做是一次卷积计算，它也是有过滤器大小和步长、padding这些参数的，不同的是，过滤器不是进行矩阵乘法运算，而是基于特定的规则（比卷积计算更简单了），最大池化，顾名思义就是取矩阵的最大值来代表这个矩阵，除了最大池化，还有平均池化函数，就是求每个矩阵的平均值作为矩阵的大小：`class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)`  
$$out(N_i, C_j,k)=max^{kH-1}_{m=0}max^{kW-1}_{m=0}input(N_{i},C_j,stride[0]h+m,stride[1]w+n)$$

### （5）




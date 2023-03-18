# dian团队春招学习记录

==**学习笔记**==

>U202215210
>
>王硕霆

## level 0

### 一.github内容

#### 1.1 理论基础

##### 1.1.1 配置

git config --list   查看配置信息

##### 1.1.2 工作流程

![img](https://www.runoob.com/wp-content/uploads/2015/02/git-process.png)

##### 1.1.3 不同区域

- 当对工作区修改（或新增）的文件执行 ==**git add**== 命令时，暂存区的目录树被更新，同时工作区修改（或新增）的文件内容被写入到对象库中的一个新的对象中，而该对象的ID被记录在暂存区的文件索引中。
- 当执行提交操作**git commit**时，暂存区的目录树写到版本库（对象库）中，master 分支会做相应的更新。即 master 指向的目录树就是提交时暂存区的目录树。
- 当执行 **git reset HEAD** 命令时，暂存区的目录树会被重写，被 master 分支指向的目录树所替换，但是工作区不受影响。

#### 1.2 实际操作

##### 1.2.1 创建本地仓库

首先，在选定的文件夹下右键，选择 **git bush here** 在进入bush界面后，用**git init** 完成一个仓库的初始化，这个你打开的文件夹就是你的**git本地仓库**，可以在这个仓库下新建你自己的文件夹，并放入文件。再通过语句

```
git add 文件名
git add 文件夹名/
```

将文件夹或文件添加到暂存区内

>遇到的问题
>
>文件夹无法add进入暂存区，可以把隐藏文件打开，删掉你所创建的文件中的.git文件，这可能是你又不小心初始化了一个仓库

再使用

```
git commit -m "备注信息" 文件夹名/
git commit -m "备注信息" 文件名
```

将暂存区指定文件添加到本地仓库。

##### 1.2.2 创建远程仓库

登录GitHub，并创建仓库,再在git中使用

```
git remote add 别名 远程仓库地址
```

使用别名可以在今后的操作中跟便捷一些,git是~~不能粘贴~~（md可以右键选择paste，草率了），每次都输入那么长串的地址属实是找虐（😭

设置完别名之后，可以利用

` git remote -v`检查已存在别名。

##### 1.2.3 上传至远程仓库

在将文件夹传至暂存区，再传到本地仓库之后，通过

```
git push "远程仓库地址/别名" "分支"
```

 将本地仓库中的文件上传到远程仓库的某个分支

##### 1.2.4 从远程仓库拉取文件

` git clone` 拷贝一份远程仓库，也就是下载项目

##### 1.2.5 建立分支

```
git branch 查看已有分支
git branch 分支名   创建新分支
git checkout 分支名   切换到该分支下
git checkout -b 分支名   创建新分支并立即切换到其下
```

##### 1.2.6查看提交历史

##### 1.2.7 打上标签



#### 1.3 命令一览

![1](D:\dian春招任务\mypro\学习记录\1.png)

这里是图片1

### 二.有关AI算法的基本知识了解

#### 2.1 数据集

##### 2.1.1 基本概念

​       数据集，从字面意义上不难理解，就是一堆数据的集合。进行机器学习，人工智能训练，是万万离不开数据的。那么我先收集一组数据，例如我手头有一堆钢琴，那么我可以收集数据如下

`（颜色=黑色；音质=清脆；种类=立式），（颜色=桃木色；音质=浑浊；种类=三角）……`

每对括号内是一条记录。而这些记录的集合就可以算作数据集。其中每条记录都是关于一个事件或具体对象的描述，称为一个***示例 “instance”*** 或***样本 “sample”*** 。反应事件或对象的莫方面表现或性质，也就是每条记录中的不同元素，称为***属性（attribute）*** 或 ***特征（feature）*** 。属性上的取值，例如“黑色”，“清脆” 称为***属性值*** 。属性张成的空间称为***属性空间(attribute space)*** ***样本空间(sample space)*** 或***输入空间*** 。

​         现在，我们可以进行一些可视化的工作。依旧是上述钢琴的例子，可以以三个元素，*颜色* ，*音质* ，*种类* 为三个坐标轴，建立三维空间。以这三种属性为坐标轴，张成了一个描述钢琴的三维空间，而每一个*示例* 也就是每一条记录，称为一个***特征向量(feature vector)*** 。

​          数据集一般包括：

- 训练集（Training Set）：模型用于训练和调整模型参数。

- 验证集（Validation Set）：用来验证模型精度和调整模型超参数，选择模型。

- 测试集（Test Set）：测试模型的泛化能力，最终对模型评估。

  因为训练集和验证集是分开的，所以模型在验证集上面的精度在一定程度上可以反映模型的泛化能力。在划分验证集的时候，需要注意验证集的分布应该与测试集尽量保持一致，不然模型在验证集上的精度就失去了指导意义。

  在使用数据集训练模型之前，我们需要先将整个数据集分为训练集、验证集、测试集。训练集是用来训练模型的，通过尝试不同的方法和思路使用训练集来训练不同的模型，再通过验证集使用交叉验证来挑选最优的模型，通过不断的迭代来改善模型在验证集上的性能，最后再通过测试集来评估模型的性能。如果数据集划分的好，可以提高模型的应用速度。如果划分的不好则会大大影响模型的应用的部署，甚至可能会使得我们之后所做的工作功亏一篑。本文讨论如何通过数据集分布和数据集大小两个方面来划分数据集

 ##### 2.1.2 类型化数据集

##### 2.2.3 非类型化数据集    

#### 2.2 模型

​        对于模型，我目前的理解就是一种规律性的东西，是从大量现实实践中抽象出来的规律。在我初高中的物理竞赛生涯中是时常接触模型这个概念的。最初的时候，我以为模型就是老师们常说的题型。但后来在我学习的过程中发现，模型可以起到一种预测的功能，例如我们在理论物理的实践中，做粒子对撞实验，改变不同的变量，做上成千上万次实验，发现都满足某一种规律，或某一个公式，这个时候，我们可以说我们发现了一种规律或者说定理，这也是一种模型，通过它，我们就可以不用做实验，就可以预测今后任意一种例子的运动方式与规律。这也是在前十八年物理学深深吸引我的地方(但是现在我已经完全被人工智能迷住了)， 而AI中的模型，我认为就是一种从数据中学得的成果，一种可以预测未知的经验，也就是所谓***机器学习(machine learning)*** 和 ***“学习算法（learning algorithm）”***，当然有些地方用模型指全局性的结果，而局部性的则称为***模式 ***。

#### 2.3 优化器

优化器的主要思路就是梯度下降法

在深度学习的反向过程中，指引损失函数的各个参数往正确的方向更新适合的大小，使得更新的参数让损失函数值不断逼近全局最小。



#### 2.4 损失函数

***损失函数(loss function)*** 或***代价函数(cost function)*** 是将随机事件或其有关变量的取值映射为非负实数以表示该随机事件的 *风险* 或 *损失* 的函数 。通过最小化损失函数以达到全局最优解。当然损失函数只是描述训练单个样本的表现，只适用与单个样本的表现。

而要衡量所有样本，要使用***成本函数(cost function)***。

以训练**logistics模型** 为例子

> 损失函数只适用于单个的训练样本。而成本函数基于参数的总成本，所以在训练logistic回归模型时，我们要找到合适的参数w和b，让成本函数尽可能的小

#### #学习笔记

![2](D:\dian春招任务\mypro\学习记录\2.jpg)

![3](D:\dian春招任务\mypro\学习记录\3.jpg)

![4](D:\dian春招任务\mypro\学习记录\4.jpg)

（ps：这里分别为图片2，3，4，我也放到文件夹中了)

## level1

### 1 准备工作（一些环境配置和术语的定义）

#### 1.0 总述

首先，要完成minist手写数据集，肯定离不开numpy，pytorch。在pycharm屡屡报错的情况下，我这里选择的方法是使用anaconda+VScode完成任务（新手友好，傻瓜操作）。 在下完anaconda之后，通过anaconda prompt导入pytorch，再在vscode中添加相关环境，我还添加了jupyter，方便调试。

这里cnn先放了一些从网上学来的代码试试

#### 1.1 池化

池化的意义在于特征降维，池化技术大大降低了对于计算资源的损耗，除此以外还有降低模型过拟合的优点。池化的思想来源于图像特征聚合统计，通俗理解就是池化虽然会使得图像变得模糊但不影响图像的辨认跟位置判断；池化还有一个优点就是平移不变性，即如果物体在图像中发生一个较小的平移（不超过感受野）,那么这样的位移并不会影像池化的效果，从而不会对模型的特征图提取发生影响。

#### 1.2 张量

**Tensor**可以将其理解为多维数组，其可以具有任意多的维度，不同**Tensor**可以有不同的**数据类型** (dtype) 和**形状** (shape)。

#### 1.3 dataloader(数据加载器类)

- Dataset 提供整个数据集的随机访问功能，每次调用都返回单个对象，例如一张图片和对应 target 等等
- Sampler 提供整个数据集随机访问的索引列表，每次调用都返回所有列表中的单个索引，常用子类是 SequentialSampler 用于提供顺序输出的索引 和 RandomSampler 用于提供随机输出的索引
- BatchSampler 内部调用 Sampler 实例，输出指定 `batch_size` 个索引，然后将索引作用于 Dataset 上从而输出 `batch_size` 个数据对象，例如 batch 张图片和 batch 个 target
- collate_fn 用于将 batch 个数据对象在 batch 维度进行聚合，生成 (b,...) 格式的数据输出，如果待聚合对象是 numpy，则会自动转化为 tensor，此时就可以输入到网络中了
- shuffle 打乱顺序

```python
torch.utils.data.DataLoader
dataset:传入dataset
batch_size:批次大小
shuffle：是否打乱
num woeker：线程数
```



#### 1.4 损失函数

这里选择的是交叉熵CosineEmbeddingLoss，具体过程就是softmax+log+nlloss

#### 1.5 优化器选择

**Adam**

Adam 算法的提出者描述其为两种随机梯度下降扩展式的优点集合，即：

- 适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。

- 均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。

  

  Adam 算法同时获得了 AdaGrad 和 RMSProp 算法的优点。Adam 不仅如 RMSProp 算法那样基于一阶矩均值计算适应性参数学习率，它同时还充分利用了梯度的二阶矩均值（即有偏方差/uncentered variance）。具体来说，算法计算了梯度的指数移动均值（exponential moving average），超参数 beta1 和 beta2 控制了这些移动均值的衰减率。

#### 1.6 超参数

超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。通常情况下，需要对超参数进行优化，给[学习机](https://baike.sogou.com/lemma/ShowInnerLink.htm?lemmaId=2099387&ss_c=ssc.citiao.link)选择一组最优超参数，以提高学习的性能和效果
例如

```
EPOCH = 1  # 训练的次数
BATCH_SIZE = 50 #每次的图片数量
LR = 0.001  # 学习率这里用0.001感觉比较合适
DOWNLOAD_MNIST = False  # True表示还没有下载数据集，如果数据集下载好了就写False
```

#### 1.7 全卷积神经网络

全卷积神经网络，顾名思义是该网络中全是卷积层链接，如下图：

![img](https://pic2.zhimg.com/80/v2-9c01766a9e070839ac10ff7bfdc083b1_720w.webp)

图2 FCN网络结构

该网络在前面两步跟CNN的结构是一样的，但是在CNN网络Flatten的时候，FCN网络将之换成了一个卷积核size为5x5，输出通道为50的卷积层，之后的全连接层都换成了1x1的卷积层。

#### 1.8 激活层

用来使模型可以处理非线性的情况，常用的有sigmod，reLU，tanh,这里我选的是reLU。

激活层使用ReLU激活函数。
线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元，是一种人工神经网络中常用的激活函数（activation function），通常指代以斜坡函数及其变种为代表的非线性函数。

![img](https://img-blog.csdnimg.cn/e3891b78542d420b8aa59b40f170585c.png)

#### 1.9 Conv2d与Linear

##### 1.9.1 Conv2d

**in_channels**是输入图像中的通道数，**out_channels**是卷积L产生的通道数

**处理图像时有三种可能情况：**

**1.如果图像是灰度的，则输入通道为1。**

**2.如果图像是彩色的，则输入通道为 3。**

**3.如果有额外的alpha通道，我们就有4个输入通道。**



为了计算每个卷积层的高度和宽度的输出维度，应用池化层后，需要记住这两个公式：

![img](https://pic2.zhimg.com/80/v2-462877b0980df2ea0490052099140129_720w.webp)



上面我们看到了两个公式，但通常两者的公式是相同的。这取决于填充、膨胀和内核大小。**在 CNN 中，卷积核/滤波器通常是3x3，而池化通常应用2x2窗口、步长2和无填充。**因此，对于这些值，输出的宽度和高度的公式将相同。

**在最后一个卷积层+池化层之后，一个或多个全连接层被添加到CNN架构中。**卷积层和池化层产生的输出是3维的，但全连接层需要一个一维数组。因此，我们使用以下函数将输出平面化为一维向量：

##### 1.9.2 Linear

==**torch.nn.Linear(in_features,out_features, bias=True)**==

这个就是构建线性层，以二维简单表述

**f(x1,x2)=w1\*x1+w2\*x2+b**

在这里:

- **in_features构成了每个输入样本的大小**
- **out_features构成每个输出样本的大小**

有两个主要功能可以使输出形状变平：

- image=image.view(image.size(0),-1)其中批量大小为image.size(0)。
- image=torch.flatten(image.size(0),start_dim=1)

### 2 开始实现手写数据识别

#### 2.1 数据集(MNIST)

**MNIST**数据集是机器学习领域中非常经典的一个数据集，由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。

可以直接通过如下代码下载

```
DOWNLOAD_MNIST = False  # True表示还没有下载数据集，如果数据集下载好了就写False

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray

    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False  # 表明是测试集
)
```

(ps:当然也可以直接官网下载  [官网下载地址](http://yann.lecun.com/exdb/mnist/)

##### 2.1.1 读取并处理数据

上述代码在已经下好数据的情况下就可以起到读取的作用

如何就是处理数据，这里用到了dataset和dataloader

```
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据
)
```

##在2.1的代码示例中有一段，更改了图片数据的格式，转换为了tensor类型

` transform=torchvision.transforms.ToTensor()`

#### 2.2 构建模型

卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->

卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->

展平多维的卷积成的特征图->接入全连接层(Linear)->输出 

```
class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # 输出通道
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output

```

## level 2

### level 2.1

forward的实现

#### 1.1 conv2d

```
    def conv2d(self, input:Tensor, kernel:Tensor, bias = 0, stride=1, padding=0):
         if padding > 0:
           input = F.pad(input, (padding, padding, padding, padding))
         bs,in_channels,input_h, input_w = input.shape
         out_channel, in_channel,kernel_h, kernel_w = kernel.shape
         #input = input.view(input.size(0), -1)
         #kernel = kernel.view(kernel.size(0), -1)
         output_h = (math.floor((input_h - kernel_h) / stride) + 1)
         output_w = (math.floor((input_w - kernel_w) / stride) + 1)

         if bias is None:
            bias = torch.zeros(out_channel)

    # 初始化输出矩阵
         output = torch.zeros(bs, out_channel, output_h, output_w)
         
         for ind in range(bs): #控制batch-size
          for oc in range(out_channel):   #
            for ic in range(in_channel):  #这两层是通过计算出的输出矩阵进行卷积核运动的逻辑控制
                for i in range(0, input_h - kernel_h + 1, stride): #对运动进行具体控制
                    for j in range(0, input_w - kernel_w + 1, stride):
                        region = input[ind, ic, i:i + kernel_h, j: j + kernel_w]
                        # 点乘相加
                        output[ind, oc, int(i / stride), int(j / stride)] += torch.sum(region * kernel[oc, ic])
            output[ind, oc] += bias[oc]


         return output
```



先计算出输出图片的长和宽（像素点有多少），再依据输出和卷积核大小，通过四重循环逻辑控制卷积运动并取值

#### 1.2 linear

```
self.output = torch.addmm(self.bias, input, self. weight.t())
```

构建一个线性的链接

#### 1.3 crossentropyloss

```
def __call__(self, input, target):
        self.output = 0.
        for i in range(input.shape[0]):

            numerator = torch.exp(input[i, target[i]])     # 分子
            denominator = torch.sum(torch.exp(input[i, :]))   # 分母

            # 计算单个损失
            loss = -torch.log(numerator / denominator)
           
            #print("单个损失： ",loss)

            # 损失累加
            self.output += loss

        # 整个 batch 的总损失是否要求平均
        
        self.output /= input.shape[0]

        
        return self.output
```

实际上就是softmax+log+nlloss的过程


# 一、预实验

初次尝试对CNN网络的训练，对于程序的实现以及相应的参数都不够了解。为了熟悉通过pytorch训练CNN网络的方法，同时对模型的性能有一个基本的认识，进行了预实验。

1. 模型选取

首先尝试实现最简单的LeNet5网络结构。因为是初次尝试，尽可能避免调整模型的结构，这里仅仅修改了最后一个全连接层的结构。相应地，为了匹配LeNet5的输入格式，在预处理中不得不将输入图像的分辨率降低为32×32。


2. 参数设置
```python
sample_size = 40  # 计算 loss 函数的间隔
epoch_num = 10    # epoch 总数 （将进行多组实验，分别取2 10 30 50）
batch_size = 4    # 批规模
learn_rate_value = 0.001  # 学习率
```

3. 训练结果

这里给出训练 epoch 总数分别取 10、30、50，三种情形下的训练结果。



4. 进一步实验

为了解释训练效果较差的原因，又尝试了 epoch 总数为 200，学习率为 0.001 的情形

得到如下结果：



**分析：**

结合四次实验的结果，可以发现，模型在测试集上较差的表现并不能单纯地用欠拟合或过拟合来解释。实际上，从实验结果来看，增加训练时间并没有显著地影响模型的预测能力，但loss值已经稳定在一个较小值的附近了。因此，猜测为当前的模型过于简单，提取特征进行泛化的能力不足，后续的实验需适当地改变模型的结构，从而更好地提取图像的特征。

**总结：**

在后续实验中，需要注意以下几点：

1. 应当提高模型的深度，最好采用现有的性能较好的模型进行训练
2. 预实验中，对图像数据的预处理需要调整，不应当为了适应模型降低输入分辨率，而应调整模型的前置模块以适应输入。
3. 数据集的规模较小，覆盖的情形可能也不够全面，后续还应当在预处理中引入随机翻转、随机裁剪以及随机旋转等模块。

# 二、正式实验

根据预实验得到的三个结论，重新选取了网络模型，增加了隐藏层的个数，同时修改了模型输入图片的像素大小。在正式实验中，将首先在无BN层与dropout层的情况下进行训练，得到相应的训练结果。接着，实现BN层与dropout层，然后重新对模型进行训练，分析训练结果的变化。

1. 模型选取

为了提高模型的可输入图片大小，同时提高模型的深度，参照AlexNet的结构进行了类似实现。为了满足题目的要求，模型在AlexNet的基础上进行了简化，去除了分类模块的两个dropout层。同时，为了降低模型的训练的时间成本，只保留了四个卷积层。

2. 参数设置
```python
sample_size = 40
epoch_num = 10  #  epoch 总数 （将进行多组实验，分别取2 10 30 50）
batch_size = 4
learn_rate_value = 0.002
```
3. 训练结果

这里给出训练 epoch 总数分别取 2、10、30、50，三种情形下的训练结果。

  
**分析：**

从训练loss曲线来看，30epoch时，模型已近似收敛，直到50epoch时，loss值也没有进一步下降，可知模型收敛所需的迭代次数约为30000。

从测试集的分类准确率来看，模型的准确率相比预实验有了显著的提升，这是由于提高隐藏层的数量，增强了模型的特征提取能力，同时输入图片分辨率变化为224×224，避免了预实验中的分辨率损失。

4. 进一步实验
    1. 加入BN层

基于pytorch框架，自行实现了BN层，其中γ矩阵与β矩阵利用pytorch的自动微分机制进行更新。

```python
self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)
self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
```

**加入BN层后的特征提取模块如下**

```python
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): mBN2d()  # 加入的BN层
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): mBN2d()  # 加入的BN层
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): mBN2d()  # 加入的BN层
    (10): ReLU(inplace=True)
    (11): Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): mBN2d() # 加入的BN层
    (13): ReLU(inplace=True)
    (14): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
```
**在加入BN层的基础上，重新训练模型，同时对比未加入BN层的情形，其结果如下**


**分析：**

结合loss曲线与测试集分类准确率，可以得出如下结论

    * 增大batchsize,训练loss曲线将更加平滑
    * batchsize取40时，训练初期loss值下降将十分缓慢，而加入BN层能有效改善这一情况
    * 加入BN层通常有利于提高模型的准确率，但增加batchsize则不一定
    * 加入BN层并适当地增加batchsize，能够提高模型的分类准确率

    2. 加入dropout层

在加入BN层的基础上，同样基于pytorch框架实现并加入dropout层。由于不含待训练参数，只需实现其前向传播方法即可

```python
class mDropOut1d(nn.Module):
    def __init__(self, p):
        super(mDropOut1d, self).__init__()
        self.p = p
    def forward(self, x):
        if not self.training:
            return x
        else:
            judge_Tensor = torch.rand_like(x) > self.p
            return judge_Tensor.float() * x / (1-self.p)
```
**加入dropout层后的分类模块如下**

```plain
  (classifier): Sequential(
    (0): mDropOut1d()  #加入的dropout层
    (1): Linear(in_features=4608, out_features=2048, bias=True)
    (2): ReLU(inplace=True)
    (3): mDropOut1d()  #加入的dropout层
    (4): Linear(in_features=2048, out_features=2048, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=2048, out_features=5, bias=True)
  )
```

    3. BN层与dropout层的pytorch实现补充

由于BN层与dropout层在训练与测试时需要采用不同的流程，这里利用pytorch框架下的train模式与eval模式来实现。

```plain
print(net.features._modules['1'].training)  # True
net.eval()  
print(net.features._modules['1'].training)  # False
net.train()
print(net.features._modules['1'].training)  # True
```
由上述运行结果可知，继承于nn.Moudle的成员变量Training可以通过封装后的train()方法与eval()方法进行布尔赋值。

因此，以BN层为例，就有如下的实现

```python
def batch_norm(self, x):
    if not self.training:
        测试时执行的语句
    else:
        训练时执行的语句
    return x
```
相应地，只需在进入测试阶段时，调用最外层模型的eval()方法即可

```python
net.eval()  # 设置为测试模式
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the all test images: %d %%' % (
        100 * correct / total))
```

# 三、补充实验

在正式实验的基础上，考虑进一步增加模型的深度能否改善预测准确率。

补充实验中，选取vgg16网络模型。

1. **模型选取**

由于使用的是cpu版本的pytorch，重新训练一个深层网络是不现实的，实验中将采用迁移学习的方式。补充实验将不再讨论BN层与dropout层的作用，重点关注随着深度的提升，模型在测试集上能否达到更好的预测性能。

```python
# —————————————————迁移学习vgg16模型————————————————————
net = models.vgg16_bn(pretrained=True)
# 对迁移模型进行调整
for parma in net.parameters():
    parma.requires_grad = False
net.classifier._modules['6'] = nn.Linear(4096, 5)
# ———————————————————————————————————————————————————————
```
```python
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): ReLU(inplace=True)
    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (39): ReLU(inplace=True)
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace=True)
    (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=5, bias=True)
  )
)
```
2. 参数设置
```plain
sample_size = 40
epoch_num = 2
batch_size = 4
learn_rate_value = 0.001
```
实际训练中，由于无法使用GPU进行加速，对数据集的一次遍历就需要15-20min，实验中首先尝试了两轮遍历。
3. 训练结果
 
**分析**

从训练loss曲线来看，2次epoch还不足以使模型收敛。但从测试集准确率来看，由于采用了含预置权重的vgg16网络，模型的预测表现有了显著的提升。一方面，这是由于vgg16这一经过特殊设计的深层网络结构自身的良好特性。另一方面，由于实验中的花卉数据集规模较小，而训练时采用迁移学习的方式，保留了在其他大数据集中训练得到的特征提取层权重，使得模型一开始就具备较强的提取图像特征的能力。

4. 进一步实验

由于此前实验中的loss函数并未收敛，后续又尝试花费更多的时间进行训练，结果如下：


**分析：**

尽管适当提高了训练时长，但对于vgg16模型的训练而言，仍然是不足的。考虑到在无法使用GPU的情形下，训练的时间成本过大，实验中不再对该网络进行更长时间的训练。

# **四、总结**

回顾预实验中的LeNet5，正式实验中的AlexNet，再到补充实验中的Vgg16，尽管模型的深度不断增加，训练的难度也不断提升，但预测的准确率的确得到了有效改善。


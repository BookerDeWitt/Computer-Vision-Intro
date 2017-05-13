# Common Layers in Deep Learning
| Framework | Python | 
| --- | --- | 
| Caffe | 2.7 | 
| Pytorch | 2.7 | 


## Fully Connected Layer 全连接层
| Framework | Code | 
| --- | --- | 
| Caffe | type: "InnerProduct"| 
| Pytorch | torch.nn.Linear(in_features, out_features, bias=True) | 

<p align="center"><img width="50%" src="pics/fc.png" /></p> 

```
fc1 = nn.Linear(3,4)
fc2 = nn.Linear(4,1)
```

## Dropout Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "Dropout"| 
| Pytorch | torch.nn.Dropout2d(p=0.5, inplace=False) | 

一种防止训练时过拟合的方法，在模型训练时随机让网络某些隐含层节点的权重不工作。

<p align="center"><img width="50%" src="pics/Dropout.png" /></p> 


## Convolution Layer 卷积层

### 2D Convolution

| Framework | Code | 
| --- | --- | 
| Caffe | type: "Convolution"| 
| Pytorch | torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) | 

往往在以图像为输入的网络中使用。

#### 卷积原理

当卷积层的输入维度(Channel)大于1时：

<p align="center"><img width="70%" src="gif/conv-layer-theory.gif" /></p>

```
Conv1 = nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=1)
```

#### 卷积层中stride, padding, dilation参数意义：

<table style="width:100%">
  <tr>
    <td><img src="gif/no_padding_no_strides.gif"></td>
    <td><img src="gif/arbitrary_padding_no_strides.gif"></td>
    <td><img src="gif/same_padding_no_strides.gif"></td>
    <td><img src="gif/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img src="gif/no_padding_strides.gif"></td>
    <td><img src="gif/padding_strides.gif"></td>
    <td><img src="gif/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides = 2</td>
    <td>Padding = 1, strides = 2</td>
    <td>Padding = 1, strides = 2 (odd)</td>
    <td></td>
  </tr>
  <tr>
    <td><img src="gif/dilation.gif"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, no stride, dilation = 2</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

### 3D Convolution

| Framework | Code | 
| --- | --- | 
| Caffe | type: "Convolution", | 
| Pytorch | torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) |

往往在以视频流为输入的网络中使用。

<p align="center"><img width="40%" src="pics/3Conv.png" /></p>

```
3DConv1 = nn.Conv3d( 1, n1, kernel_size=(d0, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
3DConv2 = nn.Conv3d(n1, n2, kernel_size=(d0, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
3DConv3 = nn.Conv3d(n2, n3, kernel_size=(d0, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
3DConv4 = nn.Conv3d(n3, n4, kernel_size=(d0, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
```

每个channel间参数不同，但每个channel内的视频流共享参数，如下图所示。

<p align="center"><img width="70%" src="pics/3Conv-share.png" /></p>


## Deconvolution Layer 反卷积层

| Framework | Code | 
| --- | --- | 
| Caffe | type: "Deconvolution"| 
| Pytorch | torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1) | 

卷积层的逆操作，常用于将卷积层生成的特征图upsampling和decode，如下图所示。

<p align="center"><img width="100%" src="pics/conv-deconv.png" /></p>

#### 反卷积层中stride, padding参数意义：

<table style="width:100%">
  <tr>
    <td><img src="gif/no_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/same_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img src="gif/no_padding_strides_transposed.gif"></td>
    <td><img src="gif/padding_strides_transposed.gif"></td>
    <td><img src="gif/padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
</table>

## Max Pooling Layer 池化层

| Framework | Code | 
| --- | --- | 
| Caffe | type: "Pooling" pool: MAX| 
| Pytorch | torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) | 

<p align="center"><img width="70%" src="pics/numerical_max_pooling.png" /></p>

```
Pool1 = nn.MaxPool2d(3, 1, padding=0)
```

## RoI Pooling Layer

<p align="center"><img width="60%" src="pics/roipooling1.png" /></p>

RoI Pooling层在论文[Fast R-CNN](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)中被提出，主要用于为大小不同的RoI(Region of Interest)区域提取大小相同的特征图，其主要过程为：
- 将RoI区域的坐标缩放到与特征图同一尺度，并对缩放后的坐标取整；
- 将缩放后的ROI区域分割为设定好的区域块（如7×7）；
- 对每个区域块内的特征值进行操作（一般是max pooling），并作为该区域块的最终输出。

[Implementation: RoI Pooling in Pytorch](https://discuss.pytorch.org/t/autograd-on-sampled-locations-on-feature-maps/1585/2)

## Max Unpooling Layer

| Framework | Code | 
| --- | --- | 
| Caffe | no official implementation, [Third-Party](https://github.com/HyeonwooNoh)| 
| Pytorch | torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0) | 

Max Pooling层的逆操作，其与Deconvlution层的区别如下图所示，Pooling层的输出是稀疏（sparse）的，后面往往要跟Convolution层来使特征图稠密化（dense）。

<p align="center"><img width="50%" src="pics/diff_unpooling.PNG" /></p>

## Crop Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "Crop"| 
| Pytorch | Tensor.contiguous() | 

将特征图尺寸剪裁到与参考特征图同样大小，caffe中可以使用专门的Crop层，Pytorch中直接对要剪裁的特征图tensor进行维度操作即可。

```
h[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous() #x.size[2], x.size[3] 分别为参考特征图的高和宽
```

## Concatenate Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "Concat"| 
| Pytorch | torch.cat(seq, dim=0)| 

将同样大小的特征图拼接在一起，形成新的维度。
<p align="center"><img width="50%" src="pics/Concatenate-layer.png" /></p>

## Batch Normalization Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "BatchNorm" and type: "Scale"| 
| Pytorch | torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)| 

Batch Normalization解决的是[Internal Covariate Shift](https://arxiv.org/abs/1502.03167)问题，即由于每一层的参数都在不断变化，所以输出的分布也会不断变化，造成梯度需要不断适应新的数据分布。所以，每一个mini batch里，对每个维度进行归一化:

![equation](http://www.sciweavers.org/upload/Tex2Img_1494672210/render.png)

上式中的γ和β为可学习参数。

## Reshape Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "Reshape"| 
| Pytorch | torch.view, nn.PixelShuffle | 

在不改变特征数值和特征总量的情况下改变特征图的形状：

<p align="center"><img width="50%" src="pics/reshape-layer.png" /></p>

```
x = torch.randn(L*r*r, h, w)
y = x.view(L, r*h, r*w)
or
nn.PixelShuffle(r)
```

## Softmax Layer
| Framework | Code | 
| --- | --- | 
| Caffe | type: "Softmax"| 
| Pytorch | torch.nn.Softmax | 

softmax用于多分类问题，比如0-9的数字识别，共有10个输出，而且这10个输出的概率和加起来应该为1，所以可以用一个softmax操作归一化这10个输出。进一步一般化，假如共有k个输出，softmax的假设可以形式化表示为：

<p align="center"><img width="50%" src="pics/softmax.png" /></p>

softmax层往往用于多分类问题的最终输出层，用来输出各类的概率，如下图所示：

<p align="center"><img width="50%" src="pics/softmax-in-net.png" /></p>



![equation](http://www.sciweavers.org/tex2img.php?eq=1%2Bsin%28mc%5E2%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

![equation](http://mathurl.com/5euwuy.png)

![equation](http://www.sciweavers.org/upload/Tex2Img_1494508243/render.png)


http://www.sciweavers.org/free-online-latex-equation-editor

http://mathurl.com/

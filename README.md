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

## Convolution Layers 卷积层

### 2D Convolution

往往在以图像为输入的网络中使用

| Framework | Code | 
| --- | --- | 
| Caffe | type: "Convolution"| 
| Pytorch | torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) | 



**卷积原理**

当卷积层的输入维度(Channel)大于1时：

<p align="center"><img width="70%" src="gif/conv-layer-theory.gif" /></p>

```
Conv1 = nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=1)
```

卷积层中stride, padding, dilation参数意义：

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
    <td><img src="gif/no_padding_strides.gif"></td>
    <td><img src="gif/padding_strides.gif"></td>
    <td><img src="gif/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
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
  <tr>
    <td><img src="gif/dilation.gif"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, no stride, dilation</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>


<p align="center"><img width="40%" src="pics/numerical_max_pooling_00.jpg" /></p>



h[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()
1111



<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

<p>$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$</p>

<img src="http://www.forkosh.com/mathtex.cgi? x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}">

![equation](http://www.sciweavers.org/tex2img.php?eq=1%2Bsin%28mc%5E2%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)
![equation](http://mathurl.com/5euwuy.png)
![equation](http://www.sciweavers.org/upload/Tex2Img_1494508243/render.png)


http://www.sciweavers.org/free-online-latex-equation-editor

http://mathurl.com/

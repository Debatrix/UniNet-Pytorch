# UniNet-Pytorch
An accurate and generalizable deep learning framework for iris recognition.

**参见**:  
Zijing Zhao and Ajay Kumar, "Towards More Accurate Iris Recognition Using Deeply Learned Spatially Corresponding Features", Internation Conference on Computer Vision (ICCV), Spotlight, Venice, Italy, 2017.  
 - [论文](http://www.comp.polyu.edu.hk/~csajaykr/deepiris.htm)
 - [代码](http://www.comp.polyu.edu.hk/~csajaykr/deepiris.htm)

## Install
- Python 3.6
- Pytorch 1.0
- torchvision 0.2.2
- opencv 3.4
- caffe(可选,用于模型转换)
- tqdm(可选,看着舒服)

## Code structure
- *ICCV17_release*
  - 论文附带的源代码与caffe模型
- *models*
   - 转换得到的Pytorch模型,其中将原论文中提到的FeatNet与MaskNet分开保存
- *util*
  - **caffemodel2pth.py**
    - 将caffemodel中的网络参数转存为pth格式,可被pytorch加载
  - **normalize.py**
    - 虹膜图像归一化
  - **segment.py**
    - 虹膜图像分割
  - **hamming.py*
    - 计算特征hamming距离
- **dataset.py**
  - 数据集文件夹格式如下:  
      - *数据集根目录*  
        - *数据集*  
          - *Image*  
            - 原始图像
          - *NormIm*  
            - 归一化图像
          - *ImParam.txt*  
            - 保存图像均值\方差
          - *SegResult.txt*  
            - 定位结果
          - *test.txt*  
            - 测试数据 每一行如:030002006R_0000000000001.bmp 2006R
          - *train.txt*  
            - 训练数据 每一行如:030002006R_0000000000001.bmp 2006R
- **extraction.py**
  - 特征提取 可通过修改LoadConfig类中的变量确定默认参数,也可以通过命令行修改参数  save参数可选'pth'/'pic' 将feature保存为数组/图片

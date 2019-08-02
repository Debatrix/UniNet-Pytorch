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
   - **normalize_tool.py**  
     - 虹膜图像归一化工具
     - 左键标注, 右键画圆(至少3个点), 'q'键确认, 其他键取消
     - 先虹膜, 后瞳孔
  - **segment.py**
    - 虹膜图像分割
  - **hamming.py*
    - 计算特征hamming距离
- **enroll_dataset.py**
    - 注册整个文件夹中的图像
- **enroll_single.py**
    - 注册单个的图像
- **evaluation.py**
    - 评估用代码
    - *TODO:代码性能测试*
- **match.py**
    - 比对代码
    - *TODO:速度极慢, python的多线程无用,需要设计成batch的*
    - *TODO:代码太老了*
- **verify.py**
    - 识别代码, 将提取得到的mat与文件夹里的全部mat比对

  
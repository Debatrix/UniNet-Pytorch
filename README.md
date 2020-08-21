# UniNet-Pytorch
An accurate and generalizable deep learning framework for iris recognition.

**Reference**:  
Zijing Zhao and Ajay Kumar, "Towards More Accurate Iris Recognition Using Deeply Learned Spatially Corresponding Features", Internation Conference on Computer Vision (ICCV), Spotlight, Venice, Italy, 2017.  
 - [论文](https://www4.comp.polyu.edu.hk/~csajaykr/myhome/papers/ICCV2017.pdf)
 - [代码](http://www.comp.polyu.edu.hk/~csajaykr/deepiris.htm)

## Install
- Python 3.6
- Pytorch 1.0+
- torchvision 0.2.2+
- opencv 3.4
- caffe(Optional)
- tqdm(Optional)

## Code structure
- *ICCV17_release*
  - 论文附带的源代码与caffe模型
- *models*
   - Source code and caffe model attached to the paper
- *util*
  - **caffemodel2pth.py**
    - Export the network parameters from caffemodel to pytorch pth format
  - **normalize.py**
    - Function of iris image normalization.
   - **normalize_tool.py**  
     - Tool for iris normalization.
     - Left click to mark, right click to draw a circle (at least 3 points),'q' key to confirm, other keys to cancel
     - Iris first, pupil rear
  - **segment.py**
    - Iris image segmentation
- **enroll_dataset.py**
    - Register all images in the folder
- **enroll_single.py**
    - Register single image in the folder
- **evaluation.py**
    - Evaluation
- **match.py**
    - Match
- **verify.py**
    - Identify
    - Compare the extracted mat file with all mat files in the folder

  

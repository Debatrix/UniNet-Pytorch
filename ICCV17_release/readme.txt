 This package is provided for re-producing experimental results in the paper:
 Zijing Zhao and Ajay Kumar, "Towards More Accurate Iris Recognition Using Deeply Learned Spatially Corresponding Features", Internation Conference on Computer Vision (ICCV), Spotlight, Venice, Italy, 2017.
 
 Anyone is permitted to use, distribute and change this program for any non-commercial usage. However each of such usage and/or publication must include above citation of this paper.
 
 -----------------------------------------------------
 
 The organization of this package is as follows:
 
 data: This folder contains a few sample iris images (raw eye images) for demonstration purpose, and the detected circle positions of all test images using method [10] for normalization of irises.
 
 matlab: This folder contains sample Matlab codes for re-producing the experimental results. 
 Enter this package in Matlab, add "matlab" folder to the path, then run demo.m to normalize iris images, extract features and masks, and match them with all-to-all protocol. 
 To run the feature extraction process, it is required to have Caffe and its Matlab interface (matcaffe) installed and configured on the system. Please follow the official instructions for setting up Caffe: http://caffe.berkeleyvision.org/installation.html
 
 models: Trained Caffe models for extracting the iris features and masks for re-producing the experimental results.
 UniNet_ND.caffemodel: model trained on ND-IRIS-0405 database, which is used in the "CrossDB" protocol for the other three databases.
 UniNet_CASIA.caffemodel: model trained on ND-IRIS-0405 and finetuned on CASIA.v4-distance database.
 UniNet_IITD.caffemodel: model trained on ND-IRIS-0405 and finetuned on IITD database.
 UniNet_WVU.caffemodel: model trained on ND-IRIS-0405 and finetuned on WVU Non-ideal database.
 UniNet_deploy.prototxt: prototxt file defining the network structure.
 
 -----------------------------------------------------
 
 Disclaimer: This package is only provided on "as it is" basis and does not include any warranty of any kind.
 
 Authors: Zijing Zhao (jason.zhao@connect.polyu.hk), Ajay Kumar (ajay.kumar@polyu.edu.hk)
	  Department of Computing, The Hong Kong Polytechnic University
 
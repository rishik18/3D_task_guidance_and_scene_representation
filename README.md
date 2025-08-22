# 3D task guidance and scene representation toolkit
### Hrishikesh Kanade, Juan Carlos Dibene Simental, Enrique Dunn
### Stevens Institute of Technology
![image](https://github.com/user-attachments/assets/e5941ff2-b54a-4803-a0ae-a8204046d9be)

## Demos
1) RGBD Demo with Realsense - demo_pcd.py
2) RGB only demo using webcam - realtime_state_change_demo.py

## For single frame fitting
1) Update the "frame" file name and change the "azure_keypoints_sample" keypoints in azure_CLIFF_single_frame.py
2) Run azure_CLIFF_single_frame.py.

The input image should be 1920x1080 as the keypoints are for that image size.

## Resizing
You can resize the RGB frames to 1920x1080 by using the resize.py file.

## State change
1) Run state_change_single_frame.py
2)  Press numbers 1 – 4 to cycle through the four sample states.

## Note for using CLIFF demo_fit
You will need to modify the smplify.py file to use the original ignore list instead of the updated one.

## Installation
Please follow the installation instructions at https://github.com/haofanwang/CLIFF for setting up CLIFF and associated files.

## Acknowledgements
1) Thank you to Dr. Nikhil Krishnaswamy and CSU’s SIGNAL Lab for the data supporting this solution.
2) This repo is built on https://github.com/haofanwang/CLIFF and https://github.com/huawei-noah/noah-research/tree/master/CLIFF.

## Contact details
Rishi (Hrishikesh) Kanade: rishikanade@outlook.com
***

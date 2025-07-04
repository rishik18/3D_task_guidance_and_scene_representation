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

## Acknowledgements
Thank you to Dr. Nikhil Krishnaswamy and CSU’s SIGNAL Lab for the data supporting this solution.
***
### This repo is a fork of https://github.com/haofanwang/CLIFF. It aims to add realtime SMPL fitting using CLIFF and Kinect pose data, and serve as a task guidance toolkit. The contents of the README below this section have been borrowed from the CLIFF repo to aid installation and setup.

# CLIFF [ECCV 2022 Oral]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cliff-carrying-location-information-in-full/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=cliff-carrying-location-information-in-full)

<p float="left">
   <img src="https://github.com/huawei-noah/noah-research/blob/master/CLIFF/assets/teaser.gif" width="100%">
</p>


## Introduction
This repo is highly built on the official [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) and contains an inference demo, and further adds accurate detector and multi-person tracker. For post-processing, motion interpolation and smooth are supported for better visualization results.


[**CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation**](
    https://arxiv.org/abs/2208.00571
).

## Major features
- [x] **[08/20/22]** Support YOLOX as single-person detector, better performance on single frame.
- [x] **[08/20/22]** Support ByteTrack as multi-person tracker, better performance on person Re-ID.
- [x] **[08/20/22]** Support linear interpolation as motion completion method, especially for occlusion.
- [x] **[08/20/22]** Support Smooth-Net as post-processing motion smooth for decresing motion jittering.
- [x] **[09/29/22]** Support SMPLify fitting given GT/Pred 2D keypoints for improving the quality of estimated SMPL params.
- [x] **[01/31/23]** Further support motion smooth for SMPL pose and translation besides of 3D joints.

## Preparation
```bash
conda create -n cliff python=3.10
pip install -r requirements.txt
```

1. Download [the SMPL models](https://smpl.is.tue.mpg.de) for rendering the reconstructed meshes
2. Download the pretrained checkpoints to run the demo [[Google Drive](
    https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP?usp=sharing)]
3. Install MMDetection and download [the pretrained checkpoints](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox)
4. Install MMTracking and download [the pretrained checkpoints](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack)

Finally put these data following the directory structure as below:
```
${ROOT}
|-- data
    smpl_mean_params.npz
    |-- ckpt
        |-- hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
        |-- res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt
        |-- hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
|-- mmdetection
    |-- checkpoints
        |-- yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
|-- mmtracking
    |-- checkpoints
        |-- bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```

## Demo

We provide demos for single-person and multi-person video.
### Single-person
Run the following command to test CLIFF on a single-person video:
```
python demo.py --ckpt data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt \
               --backbone hr48 \
               --input_path test_samples/01e222117f63f874010370037f551497ac_258.mp4 \
               --input_type video \
               --save_results \
               --make_video \
               --frame_rate 30
```
### Multi-person
Use the `--multi` flag to support multi-person tracking, `--infill` flag to support motion infill, `--smooth` flag to support motion smooth. Run the following command to test CLIFF on a multi-person video with post-processing:
```
python demo.py --ckpt data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt \
               --backbone hr48 \
               --input_path test_samples/62883594000000000102c16c.mp4 \
               --input_type video \
               --multi \
               --infill \
               --smooth \
               --save_results \
               --make_video \
               --frame_rate 30
```

## SMPLify Fitting

As the same as [SPIN](https://github.com/nkolot/SPIN), we apply SMPLify fitting after CLIFF, OpenPose format 2D Keypoints are required for convinence.

```
python3 demo_fit.py --img=examples/im1010.jpg \ 
                    --openpose=examples/im1010_openpose.json
```

## Citing the original CLIFF paper
```
@Inproceedings{li2022cliff,
  Title     = {CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation},
  Author    = {Li, Zhihao and Liu, Jianzhuang and Zhang, Zhensong and Xu, Songcen and Yan, Youliang},
  Booktitle = {ECCV},
  Year      = {2022}
}
```

## Author for the CLIFF repo
haofanwang.ai@gmail.com.

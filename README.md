# Real-Time Neural Character Rendering with Pose-Guided Multiplane Images (ECCV 2022) 

### [Project Page](https://ken-ouyang.github.io/cmpi/index.html) | [Video](https://www.youtube.com/watch?v=otL3UAak7wQ) | [Paper](https://arxiv.org/abs/2204.11820) | [Dynamic MVS Data]()


[![PGMPI](https://imgur.com/e7e4VRg.png)](https://i.imgur.com/e7e4VRg.png)

We propose pose-guided multiplane image (MPI) synthesis which can render an animatable character in real scenes with photorealistic quality. We use a portable camera rig to capture the multi-view images along with the driving signal for the moving subject. Our method generalizes the image-to-image translation paradigm, which translates the human pose to a 3D scene representation --- MPIs that can be rendered in free viewpoints, using the multi-views captures as supervision. To fully cultivate the potential of MPI, we propose depth-adaptive MPI which can be learned using variable exposure images while being robust to inaccurate camera registration. Our method demonstrates advantageous novel-view synthesis quality over the state-of-the-art approaches for characters with challenging motions. Moreover, the proposed method is generalizable to novel combinations of training poses and can be explicitly controlled. Our method achieves such expressive and animatable character rendering all in real time, serving as a promising solution for practical applications. 
## Getting started
Set up the conda environment:
```shell
conda env create -f environment.yml
conda activate pgmpi 
```
Prepare the data: **Download** the [sample scene](https://hkustconnect-my.sharepoint.com/:u:/g/personal/houyangab_connect_ust_hk/Ebd2kZRZL7xDo-YilSyIIB4BSkYEG6DjqIVUMYUZ0N6CDg?e=53REQe) and unzip it in the project folder. Start training:
```shell
sh train.sh
```
Use tensorboard to supervise the training:
```shell
tensorboard --logdir runs/
```
To evaluate or generate MVS videos:
```shell
sh evaluate.sh
```


## Data preprocess 
We provide a simple data preprocess scripts for the aligned videos. Extra packages needs to be installed. 

```shell
pip install opencv-python==4.5.5.64 
pip install mediapipe
```


To estimate the camera pose, [colmap](https://colmap.github.io/) needs to be installed. Please refer to the utils/colmap_runner.py for the estimation settings. Feel free to adjust the preprocess to your need. 


```shell
python preprocess_multi.py
```
## Training and evaluation:

We provide several training options, including the experiment path, image resolution and so on. To enable the learnable appearance and learnable depth use the following options: 
```shell
-use_appearance
-use_learnable_planes
```

We provide different rendering settings. We can render fixed view videos or bulletpoint time videos by enable the following options:
```shell
-render_fixed_view
-render_bullet_time
```
We can also render the estimated depth image by:
```shell
-render_depth
```



## Acknowlegement   
We thank the authors of [Nex-MPI](https://github.com/nex-mpi/nex-code) for sharing their code.


## Citation

```
@article{ouyang2022real,
  title={Real-Time Neural Character Rendering with Pose-Guided Multiplane Images},
  author={Ouyang, Hao and Zhang, Bo and Zhang, Pan and Yang, Hao and Yang, Jiaolong and Chen, Dong and Chen, Qifeng and Wen, Fang},
  journal={arXiv preprint arXiv:2204.11820},
  year={2022}
}
```

<div align="center">
<h2>Navigating Large-Pose Challenge for High-Fidelity Face Reenactment with Video Diffusion Model
</h2>

[Mingtao Guo]()<sup>1</sup>&nbsp;
[Guanyu Xing]()<sup>2</sup>&nbsp;
[Yanci Zhang]()<sup>3</sup>&nbsp;
[Yanli Liu]()<sup>1,3</sup>&nbsp;


<sup>1</sup> National Key Laboratory of Fundamental Science on Synthetic Vision,
Sichuan University, Chengdu, China 

<sup>2</sup> School of Cyber Science and Engineering, 
Sichuan University, Chengdu, China 

<sup>3</sup> College of Computer Science, Sichuan University, Chengdu, China 

<h3 style="color:#b22222;"> To Appear at CVPR 2025 </h3>

<h4>
<a href="">📺 Video</a>
</h4>

</div>

<div align="center">
<img src="assets/intro.png?raw=true" width="100%">
</div>
</div>

## :bookmark_tabs: Todos
We are going to make all the following contents available:
- [x] Model inference code
- [x] Model checkpoint
- [x] Training code
- [x] Data processing code

## Installation

1. Clone this repo locally:

```bash
git clone https://github.com/MingtaoGuo/Face-Reenactment-Video-Diffusion
cd Face-Reenactment-Video-Diffusion
```
2. Install the dependencies:

```bash
conda create -n relipa python=3.8
conda activate frvd
```
3. Install packages for inference:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.2.2cu121

pip install -r requirements.txt
```
## Download weights
```shell
mkdir pretrained_weights
mkdir pretrained_weights/frvd
git-lfs install

git clone https://huggingface.co/MartinGuo/Face-Reenactment-Video-Diffusion
mv Face-Reenactment-Video-Diffusion/head_embedder.pth pretrained_weights/checkpoint-30000-14frames
mv Face-Reenactment-Video-Diffusion/warping_feature_mapper.pth pretrained_weights/checkpoint-30000-14frames

mv Face-Reenactment-Video-Diffusion/insightface pretrained_weights/facecropper
mv Face-Reenactment-Video-Diffusion/landmark.onnx pretrained_weights/facecropper

mv Face-Reenactment-Video-Diffusion/appearance_feature_extractor.pth pretrained_weights/liveportrait
mv Face-Reenactment-Video-Diffusion/motion_extractor.pth pretrained_weights/liveportrait
mv Face-Reenactment-Video-Diffusion/spade_generator.pth pretrained_weights/liveportrait
mv Face-Reenactment-Video-Diffusion/warping_module.pth pretrained_weights/liveportrait

git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
mv stable-video-diffusion-img2vid-xt pretrained_weights

git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
mv sd-vae-ft-mse pretrained_weights/stable-video-diffusion-img2vid-xt
```
The weights will be put in the `./pretrained_weights` directory. Heads up! The whole downloading process could take quite a long time.
Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- relipa
|   |-- unet.pth
|   |-- ref_embedder.pth
|   |-- light_embedder.pth
|   |-- head_embedder.pth
|-- stable-video-diffusion-img2vid-xt
    |-- sd-vae-ft-mse
    |   |-- config.json
    |   |-- diffusion_pytorch_model.bin
    |-- feature_extractor
    |   |-- preprocessor_config.json
    |-- scheduler
    |   |-- scheduler_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   |-- diffusion_pytorch_model.safetensors
    |   |-- diffusion_pytorch_model.fp16.safetensors
    |-- image_encoder
    |   |-- config.json
    |   |-- model.safetensors
    |   |-- model.fp16.safetensors
```
# 🚀 Training and Inference 

## Inference of the Relightable Portrait Animation

Here's the command to run preprocess scripts: Use DECA to extract the pose from the driving video and the mesh from the reference portrait, then render shading hints by combining the driving video's pose, the reference portrait's mesh, and the target lighting.

```shell
python preprocess.py --video_path resources/WDA_DebbieDingell1_000.mp4 --source_path resources/reference.png --light_path resources/target_lighting1.png --save_path resources/shading.mp4 --motion_align relative
```

After running ```preprocess.py``` you'll get the results: 

1. Reference, 2. Mask, 3. Driving image, 4. Landmark, 5. Shading hints 
![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/shading.png)

Here's the command to run inference scripts: Guide our model with the shading hints obtained from preprocessing to generate results where the pose is consistent with that of the driving video, the identity is consistent with the reference image, and the lighting is consistent with the target lighting. 

```shell
python inference.py --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt --checkpoint_path pretrained_weights/relipa/ --video_path resources/shading.mp4 --save_path result.mp4 --guidance 4.5 --inference_steps 25 --driving_mode relighting
```

After running ```inference.py``` you'll get the results: 

1. Reference, 2. Shading hints, 3. Relighting result, 4. Driving image
![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/relighting.png)
## Training of the Relightable Portrait Animation 
```shell
python train.py --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt \
                --height 512 --width 512  --num_frames 16 --validation_steps 100 --max_train_steps 30000 \
                --gradient_accumulation_steps 8 --gradient_checkpointing True --learning_rate 1e-5 --use_8bit_adam True \
                --sample_rate 4 --num_workers 2 --checkpointing_steps 1000 --checkpoints_total_limit 2 \
                --data_meta_path TalkingHeadVideo/VFHQ/VFHQ-data-consistent.json
```
## Talking Head Video Dataset
|VFHQ-video|VFHQ-kpmap|VFHQ-mesh|VFHQ-mask|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/video.png)|![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/kpmap.png)|![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/mesh.png)|![](https://github.com/MingtaoGuo/Relightable-Portrait-Animation/blob/main/assets/maskdd.png)|

Making a training dataset for VFHQ
```text
    |-- VFHQ
        |-- Clip+zZEv-ATOpoY+P0+C2+F3168-3532_10369.mp4
        ...
```
```shell
python make_dataset.py --video_path ./VFHQ --save_path ./TalkingHeadVideo --dataset VFHQ
```
Making a training dataset for CelebV-HQ
```text
    |-- CelebV-HQ
        |-- __lRwnjxeCg_1.mp4
        ...
```
```shell
python make_dataset.py --video_path ./CelebV-HQ --save_path ./TalkingHeadVideo --dataset CelebV-HQ
```
Final dataset format
```text
./TalkingHeadVideo/
    |-- VFHQ
        |-- VFHQ-mask
            |-- Clip+zZEv-ATOpoY+P0+C2+F3168-3532_10369.mp4
             ...
        |-- VFHQ-kpmap
        |-- VFHQ-video
        |-- VFHQ-mesh
        VFHQ-data-consistent.json
    |-- CelebV-HQ
        |-- CelebV-HQ-mask
            |-- __lRwnjxeCg_1.mp4
             ...
        |-- CelebV-HQ-kpmap
        |-- CelebV-HQ-video
        |-- CelebV-HQ-mesh
        CelebV-HQ-data-consistent.json
```
# Acknowledgements
We first thank to the contributors to the [StableVideoDiffusion](https://github.com/Stability-AI/generative-models), [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [Echomimic](https://github.com/antgroup/echomimic) and [MimicMotion](https://github.com/Tencent/MimicMotion) repositories, for their open research and exploration. Furthermore, our repo incorporates some codes from [DECA](https://github.com/yfeng95/DECA), [MediaPipe](https://github.com/google-ai-edge/mediapipe) and [U2Net](https://github.com/xuebinqin/U-2-Net), and we extend our thanks to them as well.
## Citation
If you use this model in your research, please consider citing:

```bibtex
@InProceedings{Guo_2025_CVPR,
    author    = {Guo, Mingtao and Xing, Guanyu and Liu, Yanli},
    title     = {High-Fidelity Relightable Monocular Portrait Animation with Lighting-Controllable Video Diffusion Model},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {228-238}
}
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

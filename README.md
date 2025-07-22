<div align="center">
<h2>Navigating Large-Pose Challenge for High-Fidelity Face Reenactment with Video Diffusion Model
</h2>

[Mingtao Guo](https://github.com/MingtaoGuo)<sup>1</sup>&nbsp;
[Guanyu Xing](https://ccs.scu.edu.cn/info/1053/3845.htm)<sup>2</sup>&nbsp;
[Yanci Zhang](https://cs.scu.edu.cn/info/1279/13679.htm)<sup>3</sup>&nbsp;
[Yanli Liu](https://cs.scu.edu.cn/info/1279/13675.htm)<sup>1,3</sup>&nbsp;


<sup>1</sup> National Key Laboratory of Fundamental Science on Synthetic Vision,
Sichuan University, Chengdu, China 

<sup>2</sup> School of Cyber Science and Engineering, 
Sichuan University, Chengdu, China 

<sup>3</sup> College of Computer Science, Sichuan University, Chengdu, China 

<h3 style="color:#b22222;"> Accepted to CAD/Graphics 2025 and Recommended to Computers & Graphics Journal </h3>

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

## Installation

1. Clone this repo locally:

```bash
git clone https://github.com/MingtaoGuo/Face-Reenactment-Video-Diffusion
cd Face-Reenactment-Video-Diffusion
```
2. Install the dependencies:

```bash
conda create -n frvd python=3.8
conda activate frvd
```
3. Install packages for inference:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
## Download weights
```shell
mkdir pretrained_weights
mkdir pretrained_weights/checkpoint-30000-14frames
mkdir pretrained_weights/facecropper
mkdir pretrained_weights/liveportrait
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

The weights will be saved in the `./pretrained_weights`  directory. Please note that the download process may take a significant amount of time.
Once completed, the weights should be arranged in the following structure:

```text
./pretrained_weights/
|-- checkpoint-30000-14frames
|   |-- warping_feature_mapper.pth
|   |-- head_embedder.pth
|-- facecropper
|   |-- insightface
|   |-- landmark.onnx
|-- liveportrait
|   |-- appearance_feature_extractor.pth
|   |-- motion_extractor.pth
|   |-- spade_generator.pth
|   |-- warping_module.pth
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

## Inference of the FRVD

```shell
python inference.py
```

After running ```inference.py``` you'll get the results: 

1. Source image, 2. Driving video, 3. Reenactment result
![](https://github.com/MingtaoGuo/Face-Reenactment-Video-Diffusion/blob/main/assets/result.gif)
## Training of the FRVD 
```shell
python train.py 
```

# Acknowledgements
We first thank to the contributors to the [StableVideoDiffusion](https://github.com/Stability-AI/generative-models), [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend) and [MimicMotion](https://github.com/Tencent/MimicMotion) repositories, for their open research and exploration. Furthermore, our repo incorporates some codes from [LivePortrait](https://github.com/KwaiVGI/LivePortrait) and [InsightFace](https://github.com/deepinsight/insightface), and we extend our thanks to them as well.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

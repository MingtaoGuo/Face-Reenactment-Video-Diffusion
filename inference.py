import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler

from src.modules.motion_extractor import MotionExtractorEval
from src.modules.head_net import HeadNet
from src.modules.warping_feature_mapper import WarpingFeatureMapper
from src.modules.unet import UNetSpatioTemporalConditionModel
from src.pipelines.pipeline_largepose import LargePosePipeline
from src.facecropper.cropper import Cropper
from src.facecropper.crop_config import CropConfig

def export_to_video(video_frames, output_video_path, fps):
    h, w, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        frame_uint8 = (video_frames[i]).astype(np.uint8)
        img = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    video_writer.release()


class LargePoseFaceReenactment():
    def __init__(self, pretrained_model_name_or_path, motion_extractor_path, warping_feature_mapper_path, head_embedder_path, weight_dtype=torch.float16, device="cuda"):
        # Load scheduler, tokenizer and models.
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="sd-vae-ft-mse")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        warping_feature_mapper = WarpingFeatureMapper.from_unet(unet) 
        motion_extractor = MotionExtractorEval(motion_extractor_path=motion_extractor_path)
        head_embedder = HeadNet(noise_latent_channels=320)

        # Freeze vae and image_encoder
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        motion_extractor.requires_grad_(False)
        warping_feature_mapper.requires_grad_(False)
        head_embedder.requires_grad_(False)

        head_embedder.load_state_dict(torch.load(head_embedder_path))
        warping_feature_mapper.load_state_dict(torch.load(warping_feature_mapper_path))

        self.weight_dtype = weight_dtype
        self.device = device
        # Move image_encoder and vae to gpu and cast to weight_dtype
        image_encoder.to(self.device, dtype=self.weight_dtype)
        vae.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device, dtype=self.weight_dtype)
        head_embedder.to(self.device, dtype=self.weight_dtype) 
        warping_feature_mapper.to(self.device, dtype=self.weight_dtype) 
        motion_extractor.to(self.device, dtype=self.weight_dtype)

        # The models need unwrapping because for compatibility in distributed training mode.
        self.pipeline = LargePosePipeline.from_pretrained(
            pretrained_model_name_or_path,
            scheduler=noise_scheduler,
            unet=unet,
            image_encoder=image_encoder,
            vae=vae,
            head_embedder=head_embedder,
            warping_feature_mapper=warping_feature_mapper,
            motion_extractor=motion_extractor,
            torch_dtype=weight_dtype,
        )
        self.pipeline = self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=False)

    def large_pose_face_reenactment(self, ref_path, video_path, save_path, num_inference_steps=25, guidance_scale=2.5):
        path = "resources/target/"
        if not os.path.exists(path):
            os.mkdir(path)
        os.system(f"rm -r {path}/*")
        os.system(f"ffmpeg -i {video_path} {path}/%5d.png")
        
        pixel_values = []
        pixel_ref_values = np.array(Image.open(ref_path).resize([512, 512]))[..., :3]
        num = len(os.listdir(path))
        for i in range(1, num + 1):
            img = np.array(Image.open(f"{path}/{str(i).zfill(5)}.png"), dtype=np.uint8)
            img = cv2.resize(img, (512, 512))
            pixel_values.append(img[None])

        pixel_values = torch.tensor(np.concatenate(pixel_values, axis=0)[None]).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 127.5 - 1.0

        pixel_ref_values = torch.tensor(pixel_ref_values[None, None]).repeat(1, pixel_values.size(1), 1, 1, 1).to(self.device, dtype=self.weight_dtype).permute(0, 1, 4, 2, 3) / 127.5 - 1.0

        num_frames = pixel_values.size(1)
        pixel_pil = [Image.fromarray(np.uint8((pixel_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i] + 1) * 127.5)) for i in range(num_frames)]
        reference_pil = [Image.fromarray(np.uint8((pixel_ref_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, 0] + 1) * 127.5))]
        
        frames = self.pipeline(
            reference_pil, image_ref=pixel_ref_values, image_drv=pixel_values,
            num_frames=pixel_values.size(1),
            tile_size=14, tile_overlap=6,
            height=512, width=512, fps=7,
            noise_aug_strength=0.02, num_inference_steps=num_inference_steps,
            generator=None, min_guidance_scale=guidance_scale, 
            max_guidance_scale=guidance_scale, decode_chunk_size=8, output_type="pt", device="cuda"
        ).frames.cpu()
        video_frames = (frames.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).numpy()[0]

        if not os.path.exists("resources/tmp"):
            os.mkdir("resources/tmp")
        os.system("rm -r resources/tmp/*")
        for i in range(pixel_values.size(1)):
            img = video_frames[i]
            drv = np.array(pixel_pil[i])
            ref = np.array(reference_pil[0]) 
            Image.fromarray(np.uint8(np.concatenate([ref, drv, img], axis=1))).save(f"resources/tmp/{str(i).zfill(5)}.png")

        os.system(f"ffmpeg -r 24 -i resources/tmp/%05d.png -pix_fmt yuv420p -c:v libx264 {save_path} -y")

class FaceAlign():
    def __init__(self, insightface_root=None, landmark_ckpt_path=None):
        self.crop_cfg = CropConfig(insightface_root=insightface_root, 
                                   landmark_ckpt_path=landmark_ckpt_path)
        self.cropper = Cropper(crop_cfg=self.crop_cfg)

    def align(self, image_path):
        source_image = np.array(Image.open(image_path))[..., :3]
        ret_d = self.cropper.crop_source_image(source_image, self.crop_cfg)

        return ret_d["img_crop"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, default="./pretrained_weights/") 
    parser.add_argument("--source_image", type=str, default="./resources/source1.png") 
    parser.add_argument("--driving_video", type=str, default="./resources/driving1.mp4") 
    parser.add_argument("--save_path", type=str, default="./resources/result1.mp4") 
    parser.add_argument("--is_align", type=bool, default=False) 
    parser.add_argument("--num_inference_step", type=int, default=25) 
    parser.add_argument("--guidance_scale", type=int, default=2.5) 
    args = parser.parse_args()

    largepose = LargePoseFaceReenactment(pretrained_model_name_or_path=os.path.join(args.pretrained_weights, "stable-video-diffusion-img2vid-xt"), 
                                         motion_extractor_path=os.path.join(args.pretrained_weights, "liveportrait"),
                                         warping_feature_mapper_path=os.path.join(args.pretrained_weights, "checkpoint-30000-14frames/warping_feature_mapper.pth"),
                                         head_embedder_path=os.path.join(args.pretrained_weights, "checkpoint-30000-14frames/head_embedder.pth"))
    if args.is_align:
        facealign = FaceAlign(insightface_root=os.path.join(args.pretrained_weights, "facecropper/insightface"), 
                              landmark_ckpt_path=os.path.join(args.pretrained_weights, "facecropper/landmark.onnx"))
        align_source = facealign.align(args.source_image)
        Image.fromarray(np.uint8(align_source)).save("resources/align_source.png")
        source_image = "resources/align_source.png"
    else:
        source_image = args.source_image
    largepose.large_pose_face_reenactment(source_image, args.driving_video, args.save_path, 
                                          args.num_inference_step, args.guidance_scale)


        
         

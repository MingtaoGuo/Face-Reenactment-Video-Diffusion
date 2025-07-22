
import os 
import json
import random
from typing import List

import torch
import torch.nn.functional as F 
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 



class TalkingHeadVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        data_meta_paths=[],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.vid_meta = []
        for path in data_meta_paths:
            files = os.listdir(path)
            for file in files:
                if "mp4" in file:
                    self.vid_meta.append(path + file)

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_path = self.vid_meta[index]

        video_reader = VideoReader(video_path)

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read ref frame
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # read frames
        vid_pil_image_list = [ref_img]
        for index in batch_index[1:]:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )

        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_ref_img=pixel_values_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)



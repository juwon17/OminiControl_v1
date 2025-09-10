from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import os
import glob

class CustomDataset(Dataset):
    def __init__(
        self,
        image_folder: str,
        prompt: str,
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        condition_type: str = "canny",
        prompt_folder: str = None,
        target_width: int = None,
        target_height: int = None,
        condition_folder: str = None,
    ):
        self.image_folder = image_folder
        self.prompt = prompt
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.condition_type = condition_type
        self.prompt_folder = prompt_folder
        self.target_width = target_width
        self.target_height = target_height
        self.condition_folder = condition_folder

        # Get all image files from the folder
        self.image_files = []
        if image_folder.endswith(".txt"):
            with open(image_folder, 'r') as f:
                for line in f:
                    self.image_files.append(line.strip())
            random.shuffle(self.image_files)
        else:
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        self.to_tensor = T.ToTensor()

        if self.condition_folder is None:
            print("Warning: No condition folder provided. Using condition type to generate condition image.")
        if self.prompt_folder is None:
            print("Warning: No prompt folder provided. Using default prompt for all images: "+ self.prompt) 

    def __len__(self):
        return len(self.image_files)
    
    @property
    def depth_pipe(self):
        if not hasattr(self, "_depth_pipe"):
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        return self._depth_pipe

    def _get_canny_edge(self, img):
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        thres1= random.randint(30, 90)
        thres2 = random.randint(60, 150)
        low_threshold = min(thres1, thres2)
        high_threshold = max(thres1, thres2)
        edges = cv2.Canny(img_gray, low_threshold, high_threshold)

        return Image.fromarray(edges).convert("RGB")
    
    def __getitem__(self, idx):
        # Load image from file
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        # Get condition image from condition folder or generate it from condition type
        if self.condition_folder is not None:
            condition_path = os.path.join(self.condition_folder, os.path.basename(image_path))
            condition_img = Image.open(condition_path).convert("RGB")
        else:
            if self.condition_type == "canny":
                condition_img = self._get_canny_edge(image)
            elif self.condition_type == "depth":
                condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            else:
                raise ValueError(f"Invalid condition type: {self.condition_type}")

        # check condition_image and image is same size
        if condition_img.size != image.size:
            raise ValueError(f"Condition image and image are not the same size: {condition_img.size} != {image.size}")

        # if condition_img is smaller than target_size, resize it wrt original aspect ratio
        if condition_img.size[0] < self.target_width or condition_img.size[1] < self.target_height:
            if condition_img.size[0] < condition_img.size[1]:
                image = image.resize((self.target_width, self.target_height * int(image.size[1] / image.size[0])))
                condition_img = condition_img.resize((self.target_width, self.target_height * int(condition_img.size[1] / condition_img.size[0])))
            else:
                image = image.resize((self.target_width * int(image.size[0] / image.size[1]), self.target_height))
                condition_img = condition_img.resize((self.target_width * int(condition_img.size[0] / condition_img.size[1]), self.target_height))

        #random crop image and condition_img to target_size
        x1, y1 = random.randint(0, image.size[0] - self.target_width), random.randint(0, image.size[1] - self.target_height)
        x2, y2 = x1 + self.target_width, y1 + self.target_height
        
        image = image.crop((x1, y1, x2, y2))
        condition_img = condition_img.crop((x1, y1, x2, y2))

        # Get prompt from prompt folder or use default prompt
        if self.prompt_folder is not None:
            prompt_path = os.path.join(self.prompt_folder, os.path.basename(image_path).replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))
            with open(prompt_path, 'r') as f:
                description = f.read()
        else:
            description = self.prompt

        # Get the condition image
        position_delta = np.array([0, 0])

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (image.size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }

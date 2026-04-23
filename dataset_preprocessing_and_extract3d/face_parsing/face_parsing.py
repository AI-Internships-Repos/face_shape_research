import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from .bisenet import BiSeNet

class FaceParsingExtractor:
    def __init__(self, model_path: str = "face_segmentation.pth") -> None:
        model_path_obj = Path(model_path)
        if model_path_obj.is_absolute():
            self.model_path = str(model_path_obj)
        else:
            self.model_path = str((Path(__file__).resolve().parent / model_path_obj).resolve())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = None
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self._initialized = False
        self._initialize_model()
    
    def _initialize_model(self):
        if self._initialized:
            return
        
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.to(self.device)
        
        save_pth = Path(self.model_path)
   
        self.net.load_state_dict(torch.load(save_pth, map_location=self.device, weights_only=False))
        self.net.eval()
        self._initialized = True

    def extract(self, image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
        """
        Extract face parsing mask from the input image.
        Initializes model on first use (lazy loading).
        
        :param image: Input image as numpy array, PIL image, or image path.
        :return: Binary single-channel face mask as numpy array.
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = np.asarray(image)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        image = image.astype(np.uint8)

        with torch.inference_mode():
            original_size = image.shape[1], image.shape[0]
            image_resized = Image.fromarray(image).resize((512, 512), Image.Resampling.BILINEAR)
            img_tensor = self.to_tensor(image_resized)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            img_tensor = img_tensor.to(self.device)
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # Crop the excluded labels
            exclude_labels = [6, 7, 8, 9, 14, 15, 16, 17, 18]
            parsing_anno = np.where(np.isin(parsing, exclude_labels), 0, parsing)

            # binary face mask
            face_mask = (parsing_anno > 0).astype(np.uint8) * 255

            # resize back to original image size
            face_mask = cv2.resize(
                face_mask,
                original_size,
                interpolation=cv2.INTER_NEAREST
            )
        return face_mask

    def apply_mask(self, image_extract, image_original):
        """
        Apply face mask to original image (RGB output).
        :param image_extract: Binary face mask as Matlike.
        :param image_original: Original image as a numpy array.
        """
        if isinstance(image_original, Image.Image):
            original_np = np.array(image_original.convert("RGB"))
        else:
            original_np = np.asarray(image_original)

        if original_np.ndim == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        elif original_np.ndim == 3 and original_np.shape[2] == 4:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB)
        elif original_np.ndim != 3 or original_np.shape[2] != 3:
            raise ValueError(f"Unsupported image_original shape: {original_np.shape}")

        mask = np.asarray(image_extract).astype(np.uint8)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.ndim == 3 and mask.shape[2] >= 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        elif mask.ndim != 2:
            raise ValueError(f"Unsupported image_extract shape: {mask.shape}")

        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if mask.shape[:2] != original_np.shape[:2]:
            mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        result = cv2.bitwise_and(original_np, original_np, mask=mask)

        return result
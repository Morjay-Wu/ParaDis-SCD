"""Paired Synchronous Spatial and Asynchronous Photometric Transforms for SCD.

Features:
1. Synchronous Spatial Augmentations: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90, RandomAffine, Resize.
   Applied identically across (img1, img2, mask).
2. Asynchronous Photometric Variations: ColorJitter, GaussianBlur, GaussianNoise.
   Applied independently to img1 and img2 to simulate real-world illumination/weather divergence without altering mask.
3. Normalization and Tensor Conversion: ToTensor, Normalize.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter


class PairTransform:
    """Base class for paired image/mask transforms."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PairCompose(PairTransform):
    """Composes several paired transforms sequentially."""

    def __init__(self, transforms: Sequence[PairTransform]):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


# =========================================================================
# Synchronous Spatial Augmentations
# =========================================================================

class Resize(PairTransform):
    """Resizes img1, img2, and mask to target rectangular resolution (H, W)."""

    def __init__(self, size: Tuple[int, int]):
        """Args:
            size: Target (height, width) tuple.
        """
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        H, W = self.size

        # If inputs are PIL Images
        if isinstance(sample['img1'], Image.Image):
            sample['img1'] = sample['img1'].resize((W, H), resample=Image.BILINEAR)
            sample['img2'] = sample['img2'].resize((W, H), resample=Image.BILINEAR)
            if 'mask' in sample and sample['mask'] is not None:
                if isinstance(sample['mask'], Image.Image):
                    sample['mask'] = sample['mask'].resize((W, H), resample=Image.NEAREST)
        # If inputs are torch.Tensors [C, H_in, W_in]
        elif isinstance(sample['img1'], torch.Tensor):
            for k in ['img1', 'img2']:
                t = sample[k]
                if t.dim() == 3:
                    sample[k] = F.interpolate(t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                elif t.dim() == 4:
                    sample[k] = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
            if 'mask' in sample and sample['mask'] is not None:
                m = sample['mask']
                if m.dim() == 3:
                    sample['mask'] = F.interpolate(m.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
                elif m.dim() == 4:
                    sample['mask'] = F.interpolate(m, size=(H, W), mode='nearest')
        # If inputs are numpy arrays
        elif isinstance(sample['img1'], np.ndarray):
            pil1 = Image.fromarray(sample['img1'].astype(np.uint8) if sample['img1'].dtype != np.uint8 else sample['img1'])
            pil2 = Image.fromarray(sample['img2'].astype(np.uint8) if sample['img2'].dtype != np.uint8 else sample['img2'])
            sample['img1'] = np.array(pil1.resize((W, H), resample=Image.BILINEAR))
            sample['img2'] = np.array(pil2.resize((W, H), resample=Image.BILINEAR))
            if 'mask' in sample and sample['mask'] is not None:
                pil_m = Image.fromarray(sample['mask'].astype(np.uint8))
                sample['mask'] = np.array(pil_m.resize((W, H), resample=Image.NEAREST))

        return sample


class RandomHorizontalFlip(PairTransform):
    """Synchronously flips img1, img2, and mask horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            if isinstance(sample['img1'], Image.Image):
                sample['img1'] = sample['img1'].transpose(Image.FLIP_LEFT_RIGHT)
                sample['img2'] = sample['img2'].transpose(Image.FLIP_LEFT_RIGHT)
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = sample['mask'].transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(sample['img1'], torch.Tensor):
                sample['img1'] = torch.flip(sample['img1'], dims=[-1])
                sample['img2'] = torch.flip(sample['img2'], dims=[-1])
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = torch.flip(sample['mask'], dims=[-1])
            elif isinstance(sample['img1'], np.ndarray):
                sample['img1'] = np.flip(sample['img1'], axis=1).copy()
                sample['img2'] = np.flip(sample['img2'], axis=1).copy()
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = np.flip(sample['mask'], axis=1).copy()
        return sample


class RandomVerticalFlip(PairTransform):
    """Synchronously flips img1, img2, and mask vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            if isinstance(sample['img1'], Image.Image):
                sample['img1'] = sample['img1'].transpose(Image.FLIP_TOP_BOTTOM)
                sample['img2'] = sample['img2'].transpose(Image.FLIP_TOP_BOTTOM)
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = sample['mask'].transpose(Image.FLIP_TOP_BOTTOM)
            elif isinstance(sample['img1'], torch.Tensor):
                sample['img1'] = torch.flip(sample['img1'], dims=[-2])
                sample['img2'] = torch.flip(sample['img2'], dims=[-2])
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = torch.flip(sample['mask'], dims=[-2])
            elif isinstance(sample['img1'], np.ndarray):
                sample['img1'] = np.flip(sample['img1'], axis=0).copy()
                sample['img2'] = np.flip(sample['img2'], axis=0).copy()
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = np.flip(sample['mask'], axis=0).copy()
        return sample


class RandomRotation90(PairTransform):
    """Synchronously rotates img1, img2, and mask by 0, 90, 180, or 270 degrees with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # number of 90-degree rotations
            if isinstance(sample['img1'], Image.Image):
                angle = k * 90
                # PIL rotates counterclockwise
                sample['img1'] = sample['img1'].rotate(angle, expand=False, resample=Image.BILINEAR)
                sample['img2'] = sample['img2'].rotate(angle, expand=False, resample=Image.BILINEAR)
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = sample['mask'].rotate(angle, expand=False, resample=Image.NEAREST)
            elif isinstance(sample['img1'], torch.Tensor):
                sample['img1'] = torch.rot90(sample['img1'], k=k, dims=[-2, -1])
                sample['img2'] = torch.rot90(sample['img2'], k=k, dims=[-2, -1])
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = torch.rot90(sample['mask'], k=k, dims=[-2, -1])
            elif isinstance(sample['img1'], np.ndarray):
                sample['img1'] = np.rot90(sample['img1'], k=k, axes=(0, 1)).copy()
                sample['img2'] = np.rot90(sample['img2'], k=k, axes=(0, 1)).copy()
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = np.rot90(sample['mask'], k=k, axes=(0, 1)).copy()
        return sample


class RandomAffine(PairTransform):
    """Synchronous random small affine perturbation (rotation, translation, scale)."""

    def __init__(
        self,
        degrees: float = 10.0,
        translate: Tuple[float, float] = (0.05, 0.05),
        scale: Tuple[float, float] = (0.95, 1.05),
        p: float = 0.5,
    ):
        self.degrees = float(degrees)
        self.translate = translate
        self.scale = scale
        self.p = float(p)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            tx = random.uniform(-self.translate[0], self.translate[0])
            ty = random.uniform(-self.translate[1], self.translate[1])
            s = random.uniform(self.scale[0], self.scale[1])

            if isinstance(sample['img1'], torch.Tensor):
                # Construct 2D affine transformation matrix for grid_sample
                # Rotation + Scale + Translation
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad) * s, math.sin(rad) * s
                theta = torch.tensor([
                    [cos_a, -sin_a, tx],
                    [sin_a,  cos_a, ty]
                ], dtype=torch.float32, device=sample['img1'].device).unsqueeze(0)

                for k in ['img1', 'img2']:
                    t = sample[k]
                    orig_dim = t.dim()
                    if orig_dim == 3:
                        t = t.unsqueeze(0)
                    grid = F.affine_grid(theta, t.shape, align_corners=False).to(dtype=t.dtype)
                    warped = F.grid_sample(t, grid, mode='bilinear', padding_mode='border', align_corners=False)
                    sample[k] = warped.squeeze(0) if orig_dim == 3 else warped

                if 'mask' in sample and sample['mask'] is not None:
                    m = sample['mask']
                    orig_dim = m.dim()
                    if orig_dim == 3:
                        m = m.unsqueeze(0)
                    grid = F.affine_grid(theta, m.shape, align_corners=False).to(dtype=m.dtype)
                    warped_m = F.grid_sample(m, grid, mode='nearest', padding_mode='zeros', align_corners=False)
                    sample['mask'] = warped_m.squeeze(0) if orig_dim == 3 else warped_m
            elif isinstance(sample['img1'], Image.Image):
                # Apply PIL affine transform
                # Convert translation to pixels
                w, h = sample['img1'].size
                dx = tx * w
                dy = ty * h
                # PIL transform expects matrix mapping output coordinates to input coordinates
                sample['img1'] = sample['img1'].rotate(angle, translate=(dx, dy), resample=Image.BILINEAR)
                sample['img2'] = sample['img2'].rotate(angle, translate=(dx, dy), resample=Image.BILINEAR)
                if 'mask' in sample and sample['mask'] is not None:
                    sample['mask'] = sample['mask'].rotate(angle, translate=(dx, dy), resample=Image.NEAREST)
        return sample


# =========================================================================
# Asynchronous Photometric Variations (Applied Independently to img1 & img2)
# =========================================================================

class ColorJitter(PairTransform):
    """Asynchronously perturbs brightness, contrast, saturation, and hue of img1 and img2 independently."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.8,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.p = float(p)

    def _jitter_pil(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        # Brightness
        if self.brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        # Contrast
        if self.contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        # Color (Saturation)
        if self.saturation > 0:
            factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)
        return img

    def _jitter_tensor(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        # Brightness: add scalar
        if self.brightness > 0:
            b_factor = random.uniform(-self.brightness, self.brightness)
            img = img + b_factor
        # Contrast: scale around mean
        if self.contrast > 0:
            c_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            mean = img.mean(dim=[-2, -1], keepdim=True)
            img = (img - mean) * c_factor + mean
        return img.clamp(0.0, 1.0)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Asynchronously apply to img1 and img2 with independent random factors
        for k in ['img1', 'img2']:
            if isinstance(sample[k], Image.Image):
                sample[k] = self._jitter_pil(sample[k])
            elif isinstance(sample[k], torch.Tensor):
                sample[k] = self._jitter_tensor(sample[k])
        return sample


class GaussianBlur(PairTransform):
    """Asynchronously applies Gaussian blur with independent kernel/sigma to img1 and img2."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        sigma: Tuple[float, float] = (0.1, 1.5),
        p: float = 0.3,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = float(p)

    def _blur_pil(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def _blur_tensor(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            sigma_val = random.uniform(self.sigma[0], self.sigma[1])
            # Gaussian 1D kernel approximation
            k_size = self.kernel_size[0]
            radius = k_size // 2
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device)
            k1d = torch.exp(-0.5 * (x / sigma_val) ** 2)
            k1d = k1d / k1d.sum()
            k2d = torch.outer(k1d, k1d).to(dtype=img.dtype)
            k2d = k2d.expand(img.shape[0], 1, k_size, k_size)  # depthwise conv
            orig_dim = img.dim()
            inp = img.unsqueeze(0) if orig_dim == 3 else img
            blurred = F.conv2d(inp, k2d, padding=radius, groups=img.shape[0])
            return blurred.squeeze(0) if orig_dim == 3 else blurred
        return img

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in ['img1', 'img2']:
            if isinstance(sample[k], Image.Image):
                sample[k] = self._blur_pil(sample[k])
            elif isinstance(sample[k], torch.Tensor):
                sample[k] = self._blur_tensor(sample[k])
        return sample


class GaussianNoise(PairTransform):
    """Asynchronously adds zero-mean Gaussian noise independently to img1 and img2."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.02,
        p: float = 0.3,
    ):
        self.mean = float(mean)
        self.std = float(std)
        self.p = float(p)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in ['img1', 'img2']:
            if random.random() < self.p:
                if isinstance(sample[k], torch.Tensor):
                    noise = torch.randn_like(sample[k]) * self.std + self.mean
                    sample[k] = torch.clamp(sample[k] + noise, 0.0, 1.0)
                elif isinstance(sample[k], np.ndarray):
                    noise = np.random.normal(self.mean, self.std, size=sample[k].shape)
                    sample[k] = np.clip(sample[k] + noise * 255.0, 0, 255).astype(sample[k].dtype)
        return sample


# =========================================================================
# Tensor Conversion and Normalization
# =========================================================================

class ToTensor(PairTransform):
    """Converts PIL Images or NumPy arrays to float torch.Tensors in [0, 1].

    img1, img2 -> [3, H, W] in [0, 1]
    mask -> [1, H, W] in {0, 1}
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in ['img1', 'img2']:
            val = sample[k]
            if isinstance(val, Image.Image):
                arr = np.array(val)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[-1] == 4:
                    arr = arr[..., :3]
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                sample[k] = tensor
            elif isinstance(val, np.ndarray):
                if val.ndim == 2:
                    val = np.stack([val] * 3, axis=-1)
                elif val.shape[-1] == 4:
                    val = val[..., :3]
                if val.dtype == np.uint8:
                    tensor = torch.from_numpy(val).permute(2, 0, 1).float() / 255.0
                else:
                    tensor = torch.from_numpy(val).permute(2, 0, 1).float()
                sample[k] = tensor

        if 'mask' in sample and sample['mask'] is not None:
            m = sample['mask']
            if isinstance(m, Image.Image):
                m_arr = np.array(m)
                if m_arr.ndim == 3:
                    m_arr = m_arr[..., 0]
                m_tensor = (torch.from_numpy(m_arr) > 0).float().unsqueeze(0)  # [1, H, W]
                sample['mask'] = m_tensor
            elif isinstance(m, np.ndarray):
                if m.ndim == 3:
                    m = m[..., 0]
                m_tensor = (torch.from_numpy(m) > 0).float().unsqueeze(0)  # [1, H, W]
                sample['mask'] = m_tensor
            elif isinstance(m, torch.Tensor):
                if m.dim() == 2:
                    sample['mask'] = (m > 0).float().unsqueeze(0)
                elif m.dim() == 3 and m.shape[0] != 1:
                    sample['mask'] = (m[:1] > 0).float()
                else:
                    sample['mask'] = (m > 0).float()

        return sample


class Normalize(PairTransform):
    """Normalizes img1 and img2 with channel-wise mean and std (mask is untouched)."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for k in ['img1', 'img2']:
            t = sample[k]
            mean = self.mean.to(device=t.device, dtype=t.dtype)
            std = self.std.to(device=t.device, dtype=t.dtype)
            sample[k] = (t - mean) / std
        return sample


def build_scd_transforms(
    split: str = 'train',
    img_size: Tuple[int, int] = (480, 640),
    normalize: bool = False,
) -> PairCompose:
    """Builds standard paired transformation pipeline for SCD.

    Args:
        split: 'train', 'val', or 'test'.
        img_size: Target (height, width) tuple.
        normalize: Whether to apply ImageNet channel normalization.

    Returns:
        PairCompose pipeline.
    """
    transforms: List[PairTransform] = []

    # 1. Base Resize to rectangular resolution
    transforms.append(Resize(size=img_size))

    if split == 'train':
        # 2. Synchronous Spatial Augmentations (preserving horizon orientation & rectangular aspect)
        transforms.append(RandomHorizontalFlip(p=0.5))

        # 3. Asynchronous Photometric Variations
        transforms.append(ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5))

    # 4. Convert to PyTorch tensors in [0, 1]
    transforms.append(ToTensor())

    # 5. Optional normalization
    if normalize:
        transforms.append(Normalize())

    return PairCompose(transforms)

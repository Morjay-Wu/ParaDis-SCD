"""PCD Dataset: Panoramic Change Detection (TSUNAMI / GSV).

CVPR Main-Conference benchmark dataset for wide-angle cylindrical panoramic sequences
under severe camera viewpoint translation, rotation, and environmental change.
Default rectangular resolution: 224 x 1024.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import glob
import os
import numpy as np
from PIL import Image, ImageDraw

from .base import BaseSCDDataset
from .transforms import PairCompose


class PCDDataset(BaseSCDDataset):
    """PCD Panoramic Scene Change Detection Dataset (TSUNAMI & GSV).

    Args:
        root_dir: Path to PCD data folder (containing TSUNAMI or GSV subfolders).
        split: 'train', 'val', or 'test'.
        img_size: Image resolution tuple (H, W), default (224, 1024).
        transform: Optional paired transformation pipeline.
        num_synthetic_samples: Fallback sample count (default 100).
        seed: Random seed (default 42).
    """

    DEFAULT_SERVER_ROOT = '/home/WUtongquan/dataset/PCD'

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        img_size: Tuple[int, int] = (224, 1024),
        transform: Optional[Union[PairCompose, Callable]] = None,
        num_synthetic_samples: int = 100,
        seed: int = 42,
    ):
        if root_dir is None and num_synthetic_samples == 100 and os.path.isdir(self.DEFAULT_SERVER_ROOT):
            root_dir = self.DEFAULT_SERVER_ROOT
        super().__init__(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            transform=transform,
            num_synthetic_samples=num_synthetic_samples,
            seed=seed,
        )

    def _discover_real_samples(self, root_dir: str, split: str) -> List[Dict[str, Any]]:
        """Discovers real PCD panoramic pairs and masks if available on disk."""
        samples: List[Dict[str, Any]] = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

        def _collect_from_dir(base_dir: str) -> List[Dict[str, Any]]:
            found: List[Dict[str, Any]] = []
            if not os.path.isdir(base_dir):
                return found

            d0, d1, dm = None, None, None
            for cand in ('t0', 'img1', 'A', 'input1'):
                p = os.path.join(base_dir, cand)
                if os.path.isdir(p):
                    d0 = p
                    break
            for cand in ('t1', 'img2', 'B', 'input2'):
                p = os.path.join(base_dir, cand)
                if os.path.isdir(p):
                    d1 = p
                    break
            for cand in ('groundtruth', 'gt', 'mask', 'label', 'change_mask'):
                p = os.path.join(base_dir, cand)
                if os.path.isdir(p):
                    dm = p
                    break

            if d0 and d1 and dm:
                t1_map = {
                    os.path.splitext(f)[0]: os.path.join(d1, f)
                    for f in os.listdir(d1)
                    if f.lower().endswith(valid_exts)
                }
                gt_map = {
                    os.path.splitext(f)[0]: os.path.join(dm, f)
                    for f in os.listdir(dm)
                    if f.lower().endswith(valid_exts)
                }
                for f0_name in sorted(os.listdir(d0)):
                    if not f0_name.lower().endswith(valid_exts):
                        continue
                    stem = os.path.splitext(f0_name)[0]
                    if stem in t1_map and stem in gt_map:
                        found.append({
                            'img1_path': os.path.join(d0, f0_name),
                            'img2_path': t1_map[stem],
                            'mask_path': gt_map[stem],
                            'name': f"pcd_{os.path.basename(base_dir)}_{stem}",
                        })
            return found

        subsets = ['TSUNAMI', 'GSV', '']
        split_candidates: List[str] = []
        for sub in subsets:
            c1 = os.path.join(root_dir, sub, split) if sub else os.path.join(root_dir, split)
            c2 = os.path.join(root_dir, split, sub) if sub else os.path.join(root_dir, split)
            for c in (c1, c2):
                if c not in split_candidates:
                    split_candidates.append(c)

        for cand_dir in split_candidates:
            samples.extend(_collect_from_dir(cand_dir))

        if len(samples) > 0:
            return samples

        # Fallback: unsplit TSUNAMI / GSV / root_dir
        unsplit_samples: List[Dict[str, Any]] = []
        for sub in subsets:
            cand_dir = os.path.join(root_dir, sub) if sub else root_dir
            unsplit_samples.extend(_collect_from_dir(cand_dir))

        if len(unsplit_samples) > 0:
            n = len(unsplit_samples)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            if split == 'train':
                return unsplit_samples[:n_train]
            elif split == 'val':
                return unsplit_samples[n_train:n_train + n_val]
            else:
                return unsplit_samples[n_train + n_val:]

        return samples

    def _synthesize_scene_pair(
        self,
        sample_info: Dict[str, Any],
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Synthesizes a wide-aspect cylindrical panoramic scene with panoramic structures and changes."""
        rng = np.random.RandomState(sample_info['seed'])
        H, W = self.img_size

        bg_array = np.zeros((H, W, 3), dtype=np.uint8)
        horizon_y = int(H * (0.45 + rng.uniform(-0.04, 0.04)))

        # Panoramic sky gradient
        for y in range(horizon_y):
            ratio = y / max(1, horizon_y)
            sky_c = (np.array([120, 160, 220]) * (1.0 - ratio) + np.array([200, 220, 240]) * ratio).astype(np.uint8)
            bg_array[y, :] = sky_c

        # Ground / road / water base
        ground_c = rng.randint(60, 100, size=3)
        bg_array[horizon_y:, :] = ground_c

        img1_pil = Image.fromarray(bg_array)
        draw1 = ImageDraw.Draw(img1_pil)

        # Draw distant panoramic coastline / mountains / skyline along 1024 width
        mountain_pts = [(0, horizon_y)]
        for x in range(0, W + 40, 40):
            my = horizon_y - rng.randint(int(H * 0.1), int(H * 0.35))
            mountain_pts.append((x, my))
        mountain_pts.append((W, horizon_y))
        draw1.polygon(mountain_pts, fill=(90, 110, 95), outline=(50, 70, 55))

        # Panoramic wide road or shoreline
        draw1.rectangle([0, int(horizon_y + H * 0.15), W, H], fill=(50, 52, 58))
        # Continuous road edge line
        draw1.line([(0, int(horizon_y + H * 0.15)), (W, int(horizon_y + H * 0.15))], fill=(220, 220, 220), width=3)

        # Add static utility poles or trees periodically across panoramic field
        for px in range(rng.randint(20, 80), W, rng.randint(140, 220)):
            draw1.line([(px, horizon_y - int(H * 0.3)), (px, horizon_y + int(H * 0.15))], fill=(70, 50, 30), width=3)
            draw1.ellipse([px - 15, horizon_y - int(H * 0.45), px + 15, horizon_y - int(H * 0.2)], fill=(35, 95, 45))

        # Camera horizontal panoramic roll / shift (dx, dy)
        dx = rng.uniform(-0.05, 0.05) * W
        dy = rng.uniform(-0.04, 0.04) * H
        scale = rng.uniform(0.97, 1.03)
        matrix = [1.0 / scale, 0.0, -dx / scale, 0.0, 1.0 / scale, -dy / scale]
        img2_pil = img1_pil.transform((W, H), Image.AFFINE, matrix, resample=Image.BILINEAR)

        # Ambient lighting variation
        illum = rng.uniform(0.8, 1.25)
        img2_pil = Image.fromarray((np.array(img2_pil).astype(float) * illum).clip(0, 255).astype(np.uint8))

        # Ground truth panoramic changes: vehicles, construction barriers, or temporary structures
        mask_pil = Image.new('L', (W, H), 0)
        draw2 = ImageDraw.Draw(img2_pil)
        draw_mask = ImageDraw.Draw(mask_pil)

        num_changes = rng.randint(2, 5)
        for _ in range(num_changes):
            cx = rng.randint(int(W * 0.08), int(W * 0.92))
            cy = rng.randint(int(horizon_y + H * 0.2), int(H * 0.88))
            cw = rng.randint(int(W * 0.04), int(W * 0.10))
            ch = rng.randint(int(H * 0.14), int(H * 0.32))
            box = [cx - cw // 2, cy - ch, cx + cw // 2, cy]

            obj_color = tuple(rng.randint(30, 230, size=3))
            draw2.rectangle(box, fill=obj_color, outline=(20, 20, 20), width=2)
            draw_mask.rectangle(box, fill=255)

        return img1_pil, img2_pil, mask_pil

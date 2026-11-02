"""VL-CMU-CD Dataset: Street-View Scene Change Detection.

CVPR Main-Conference benchmark dataset for autonomous driving street scenes
under severe seasonal illumination shifts, weather changes, and viewpoint drift.
Default rectangular resolution: 480 x 640.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import glob
import os
import numpy as np
from PIL import Image, ImageDraw

from .base import BaseSCDDataset
from .transforms import PairCompose


class VLCMUDataset(BaseSCDDataset):
    """VL-CMU-CD Street-View Scene Change Detection Dataset.

    Args:
        root_dir: Path to VL-CMU-CD data directory.
        split: 'train', 'val', or 'test'.
        img_size: Image resolution tuple (H, W), default (480, 640).
        transform: Optional paired transformation pipeline.
        num_synthetic_samples: Fallback sample count (default 100).
        seed: Random seed (default 42).
    """

    DEFAULT_SERVER_ROOT = '/home/WUtongquan/dataset/VL-CMU-CD'

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = 'train',
        img_size: Tuple[int, int] = (480, 640),
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
        """Discovers real VL-CMU-CD image pairs and masks if available on disk."""
        samples: List[Dict[str, Any]] = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

        split_dir = os.path.join(root_dir, split)
        has_split_dir = os.path.isdir(split_dir)
        search_root = split_dir if has_split_dir else root_dir

        if not os.path.isdir(search_root):
            return samples

        # Candidate directories: search_root itself (e.g. root_dir/train/{t0,t1,gt})
        # plus any sequence subdirectories inside search_root
        candidate_dirs = [search_root] + sorted([
            os.path.join(search_root, d)
            for d in os.listdir(search_root)
            if os.path.isdir(os.path.join(search_root, d))
        ])

        folder_triplets = [
            ('t0', 't1', 'gt'),
            ('t0', 't1', 'mask'),
            ('img1', 'img2', 'mask'),
            ('img1', 'img2', 'gt'),
            ('A', 'B', 'label'),
            ('A', 'B', 'OUT'),
        ]

        for s_dir in candidate_dirs:
            for t0_name, t1_name, mask_name in folder_triplets:
                d0 = os.path.join(s_dir, t0_name)
                d1 = os.path.join(s_dir, t1_name)
                dm = os.path.join(s_dir, mask_name)
                if os.path.isdir(d0) and os.path.isdir(d1) and os.path.isdir(dm):
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
                            samples.append({
                                'img1_path': os.path.join(d0, f0_name),
                                'img2_path': t1_map[stem],
                                'mask_path': gt_map[stem],
                                'name': f"vlcmu_{os.path.basename(s_dir)}_{stem}",
                            })
                    break

        if not has_split_dir and len(samples) > 0:
            n = len(samples)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            if split == 'train':
                samples = samples[:n_train]
            elif split == 'val':
                samples = samples[n_train:n_train + n_val]
            else:
                samples = samples[n_train + n_val:]

        return samples

    def _synthesize_scene_pair(
        self,
        sample_info: Dict[str, Any],
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Synthesizes an outdoor autonomous driving street scene with road, sidewalk, buildings, and vehicles."""
        rng = np.random.RandomState(sample_info['seed'])
        H, W = self.img_size

        bg_array = np.zeros((H, W, 3), dtype=np.uint8)
        horizon_y = int(H * (0.42 + rng.uniform(-0.04, 0.04)))

        # Sky (upper area)
        sky_blue = rng.randint(180, 230, size=3) * np.array([0.7, 0.85, 1.0])
        bg_array[:horizon_y, :] = sky_blue.clip(0, 255).astype(np.uint8)

        # Distant street buildings skyline
        img1_pil = Image.fromarray(bg_array)
        draw1 = ImageDraw.Draw(img1_pil)

        b_x = 0
        while b_x < W:
            bw = rng.randint(int(W * 0.08), int(W * 0.2))
            bh = rng.randint(int(H * 0.18), int(H * 0.38))
            b_color = tuple(rng.randint(90, 160, size=3))
            draw1.rectangle([b_x, horizon_y - bh, b_x + bw, horizon_y], fill=b_color, outline=(40, 40, 40))
            # Building windows
            for wy in range(horizon_y - bh + 10, horizon_y - 10, 18):
                for wx in range(b_x + 8, b_x + bw - 8, 14):
                    draw1.rectangle([wx, wy, wx + 6, wy + 8], fill=(240, 230, 170) if rng.rand() > 0.4 else (60, 60, 60))
            b_x += bw

        # Road asphalt (lower area)
        draw1.rectangle([0, horizon_y, W, H], fill=(55, 55, 60))

        # Sidewalk on sides
        vanishing_x = int(W * (0.5 + rng.uniform(-0.1, 0.1)))
        vanishing_pt = (vanishing_x, horizon_y)

        # Road perspective curbs
        draw1.polygon([vanishing_pt, (0, H), (int(W * 0.18), H)], fill=(150, 150, 150))
        draw1.polygon([vanishing_pt, (W, H), (int(W * 0.82), H)], fill=(150, 150, 150))

        # Lane center dashed stripes
        for step in range(1, 9):
            t1 = (step / 9.0) ** 1.8
            t2 = ((step + 0.5) / 9.0) ** 1.8
            y1 = int(horizon_y + t1 * (H - horizon_y))
            y2 = int(horizon_y + t2 * (H - horizon_y))
            draw1.line([(vanishing_x, y1), (vanishing_x, y2)], fill=(240, 240, 240), width=max(1, int(t2 * 6)))

        # Static background parked car or roadside tree
        tree_x = rng.randint(10, int(W * 0.15))
        draw1.rectangle([tree_x, horizon_y - int(H * 0.15), tree_x + 8, horizon_y], fill=(90, 60, 40))
        draw1.ellipse([tree_x - 20, horizon_y - int(H * 0.3), tree_x + 28, horizon_y - int(H * 0.1)], fill=(40, 120, 50))

        # Camera translation & rotation
        dx = rng.uniform(-0.06, 0.06) * W
        dy = rng.uniform(-0.03, 0.03) * H
        scale = rng.uniform(0.96, 1.04)
        matrix = [1.0 / scale, 0.0, -dx / scale, 0.0, 1.0 / scale, -dy / scale]
        img2_pil = img1_pil.transform((W, H), Image.AFFINE, matrix, resample=Image.BILINEAR)

        # Seasonal/weather lighting shift
        weather_mode = rng.choice(['overcast', 'sunny', 'dusk'])
        arr2 = np.array(img2_pil).astype(float)
        if weather_mode == 'overcast':
            arr2 = arr2 * 0.85
        elif weather_mode == 'sunny':
            arr2 = arr2 * 1.15
        elif weather_mode == 'dusk':
            arr2 = arr2 * np.array([1.15, 0.9, 0.8])
        img2_pil = Image.fromarray(arr2.clip(0, 255).astype(np.uint8))

        # Ground truth road changes: vehicle parked/departed or pedestrian walking
        mask_pil = Image.new('L', (W, H), 0)
        draw2 = ImageDraw.Draw(img2_pil)
        draw_mask = ImageDraw.Draw(mask_pil)

        num_changes = rng.randint(1, 3)
        for _ in range(num_changes):
            cx = rng.randint(int(W * 0.22), int(W * 0.78))
            cy = rng.randint(int(horizon_y + H * 0.1), int(H * 0.88))
            cw = rng.randint(int(W * 0.12), int(W * 0.24))
            ch = rng.randint(int(H * 0.08), int(H * 0.16))
            car_box = [cx - cw // 2, cy - ch, cx + cw // 2, cy]

            car_color = tuple(rng.randint(30, 240, size=3))
            draw2.rectangle(car_box, fill=car_color, outline=(20, 20, 20), width=2)
            # Car wheels
            draw2.ellipse([car_box[0] + 6, cy - 8, car_box[0] + 20, cy + 6], fill=(15, 15, 15))
            draw2.ellipse([car_box[2] - 20, cy - 8, car_box[2] - 6, cy + 6], fill=(15, 15, 15))

            draw_mask.rectangle(car_box, fill=255)

        return img1_pil, img2_pil, mask_pil

"""ChangeSim Dataset: Warehouse Scene Change Detection.

CVPR Main-Conference benchmark dataset for indoor mobile robot inspection
under severe viewpoint variations, dynamic lighting, and warehouse clutter.
Default rectangular resolution: 480 x 640.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import glob
import os
import numpy as np
from PIL import Image, ImageDraw

from .base import BaseSCDDataset
from .transforms import PairCompose


class ChangeSimDataset(BaseSCDDataset):
    """ChangeSim Warehouse Scene Change Detection Dataset.

    Args:
        root_dir: Path to ChangeSim data folder.
        split: 'train', 'val', or 'test'.
        img_size: Image resolution tuple (H, W), default (480, 640).
        transform: Optional paired transformation pipeline.
        num_synthetic_samples: Fallback sample count (default 100).
        seed: Random seed (default 42).
    """

    DEFAULT_SERVER_ROOT = '/home/WUtongquan/dataset/ChangeSim'

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
        """Discovers real ChangeSim image pairs and masks if available on disk."""
        samples: List[Dict[str, Any]] = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

        split_dir = os.path.join(root_dir, split)
        has_split_dir = os.path.isdir(split_dir)
        search_root = split_dir if has_split_dir else root_dir

        if not os.path.isdir(search_root):
            return samples

        for curr_root, dirs, _ in os.walk(search_root):
            dirs.sort()
            d0, d1, dm = None, None, None
            for cand in ('t0_rgb', 't0', 'img1', 'A', 'reference'):
                if cand in dirs:
                    d0 = os.path.join(curr_root, cand)
                    break
            for cand in ('t1_rgb', 't1', 'img2', 'B', 'query'):
                if cand in dirs:
                    d1 = os.path.join(curr_root, cand)
                    break
            for cand in ('change_mask', 'mask', 'gt', 'label', 'semantic_segmentation'):
                if cand in dirs:
                    dm = os.path.join(curr_root, cand)
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
                        samples.append({
                            'img1_path': os.path.join(d0, f0_name),
                            'img2_path': t1_map[stem],
                            'mask_path': gt_map[stem],
                            'name': f"{os.path.basename(curr_root)}_{stem}",
                        })

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
        """Synthesizes an authentic industrial warehouse scene with racking, pallets, and lighting variation."""
        rng = np.random.RandomState(sample_info['seed'])
        H, W = self.img_size

        # Industrial warehouse base
        bg_array = np.zeros((H, W, 3), dtype=np.uint8)
        horizon_y = int(H * (0.32 + rng.uniform(-0.03, 0.03)))

        # Distant warehouse corrugated metal walls
        wall_color = rng.randint(90, 140, size=3)
        bg_array[:horizon_y, :] = wall_color

        # Floor: polished industrial concrete with yellow safety marking lines
        floor_color = rng.randint(80, 110, size=3)
        bg_array[horizon_y:, :] = floor_color

        img1_pil = Image.fromarray(bg_array)
        draw1 = ImageDraw.Draw(img1_pil)

        vanishing_x = int(W * (0.5 + rng.uniform(-0.1, 0.1)))
        vanishing_pt = (vanishing_x, horizon_y)

        # Floor lane markings (yellow stripes)
        for offset in [-0.25, 0.25]:
            fx = int(W * (0.5 + offset))
            draw1.line([vanishing_pt, (fx, H)], fill=(220, 190, 40), width=4)

        # High-bay storage racks on left and right flanks
        rack_color = (40, 70, 130)  # Industrial blue racking
        # Left rack
        draw1.rectangle([0, int(horizon_y * 0.4), int(W * 0.22), H], fill=(50, 50, 50), outline=rack_color, width=3)
        # Horizontal rack shelf beams
        for shelf_y in np.linspace(horizon_y * 0.5, H * 0.9, 5):
            draw1.line([(0, int(shelf_y)), (int(W * 0.22), int(shelf_y))], fill=rack_color, width=4)

        # Right rack
        draw1.rectangle([int(W * 0.78), int(horizon_y * 0.4), W, H], fill=(50, 50, 50), outline=rack_color, width=3)
        for shelf_y in np.linspace(horizon_y * 0.5, H * 0.9, 5):
            draw1.line([(int(W * 0.78), int(shelf_y)), (W, int(shelf_y))], fill=rack_color, width=4)

        # Static wooden pallets on floor
        for _ in range(rng.randint(2, 4)):
            px = rng.randint(int(W * 0.28), int(W * 0.72))
            py = rng.randint(int(horizon_y + H * 0.1), int(H * 0.85))
            pw, ph = rng.randint(int(W * 0.08), int(W * 0.14)), rng.randint(int(H * 0.04), int(H * 0.08))
            draw1.rectangle([px - pw // 2, py - ph, px + pw // 2, py], fill=(160, 120, 70), outline=(80, 60, 30))

        # Camera motion warp
        dx = rng.uniform(-0.05, 0.05) * W
        dy = rng.uniform(-0.03, 0.03) * H
        scale = rng.uniform(0.97, 1.03)
        matrix = [1.0 / scale, 0.0, -dx / scale, 0.0, 1.0 / scale, -dy / scale]
        img2_pil = img1_pil.transform((W, H), Image.AFFINE, matrix, resample=Image.BILINEAR)

        # Warehouse illumination shift (fluorescent flickers)
        illum = rng.uniform(0.75, 1.2)
        img2_pil = Image.fromarray((np.array(img2_pil).astype(float) * illum).clip(0, 255).astype(np.uint8))

        # Ground truth warehouse changes: shipping cartons or plastic totes placed/removed
        mask_pil = Image.new('L', (W, H), 0)
        draw2 = ImageDraw.Draw(img2_pil)
        draw_mask = ImageDraw.Draw(mask_pil)

        num_changes = rng.randint(1, 4)
        for _ in range(num_changes):
            cx = rng.randint(int(W * 0.25), int(W * 0.75))
            cy = rng.randint(int(horizon_y + H * 0.15), int(H * 0.9))
            cw = rng.randint(int(W * 0.09), int(W * 0.18))
            ch = rng.randint(int(H * 0.09), int(H * 0.22))
            box = [cx - cw // 2, cy - ch, cx + cw // 2, cy]

            box_type = rng.choice(['cardboard_box', 'plastic_tote'])
            if box_type == 'cardboard_box':
                color = tuple((rng.randint(180, 210, size=3) * np.array([1.0, 0.85, 0.6])).astype(int))
            else:
                tote_colors = [(220, 50, 40), (30, 140, 210), (40, 170, 70)]
                color = tote_colors[rng.choice(len(tote_colors))]

            draw2.rectangle(box, fill=tuple(int(c) for c in color), outline=(20, 20, 20), width=2)
            draw_mask.rectangle(box, fill=255)

        return img1_pil, img2_pil, mask_pil

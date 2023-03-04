import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.criterion import dist_between_onehot
from rotate_captcha_crack.dataset import ImgSeqFromPaths, RotDataset
from rotate_captcha_crack.model import RotNet_reg, WhereIsMyModel
from rotate_captcha_crack.utils import default_num_workers, slice_from_range

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        dataset_root = Path("./datasets/Landscape-Dataset")

        img_paths = list(dataset_root.glob('*.jpg'))
        test_img_paths = slice_from_range(img_paths, (0.95, 1.0))
        test_dataset = RotDataset(ImgSeqFromPaths(test_img_paths))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            num_workers=default_num_workers(),
            drop_last=True,
        )

        model = RotNet_reg(train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "last.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path)))
        model.to(device=device)
        model.eval()

        total_degree_diff = 0.0
        batch_count = 0

        for source, target in test_dataloader:
            source: Tensor = source.to(device=device)
            target: Tensor = target.to(device=device)

            predict: Tensor = model(source)

            digree_diff = dist_between_onehot(predict, target) * 360
            total_degree_diff += digree_diff

            batch_count += 1

        print(f"test_loss: {total_degree_diff/batch_count:.4f} degrees")

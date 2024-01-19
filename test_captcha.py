import argparse

## 单独运行的时候需要添加
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# import sys
# sys.path.append("../")
# print(sys.path)
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        model = RotNetR(train=False, cls_num=180)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")

        model.load_state_dict(torch.load(str(model_path)))
        model = model.to(device=device)
        model.eval()

        # img = Image.open("datasets/tieba/1615096444.jpg")
        # img = Image.open("datasets/tieba/1615096443.jpg")


        img = Image.open("datasets/tieba/test-1111.jpg")

        # img = Image.open("datasets/1.jpg")
        img_ts = process_captcha(img)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360
        print(f"Predict degree: {degree:.4f}°")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    print(f"111 ")
    plt.figure("debug")
    plt.imshow(img)
    print(f"22222 ")
    plt.show()

    print(f"3333 ")
    img.show()

    img.save("./test-1111-1111.jpg")

    # # 显示图像
    # cv2.imshow('Original Image', img)
    # # cv2.imshow('Rotated Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

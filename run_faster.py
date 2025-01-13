import cv2
import matplotlib
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision
from keras.src.utils.summary_utils import bold_text

from faster import FasterRCNNwithFPN
from fpn import FeaturePyramidNetwork
from resnet import resnet50
from utils import normalize_img
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models._meta import _COCO_CATEGORIES
import numpy as np


def generate_colormap(n):
    colormap = matplotlib.cm.get_cmap("jet", n)
    st0 = np.random.get_state()
    np.random.seed(44)
    perm = np.random.permutation(n)
    np.random.set_state(st0)
    cmap = np.round(colormap(np.arange(n)) * 255).astype(np.uint8)[:,:3]
    cmap = cmap[perm, :]
    return cmap


if __name__ == '__main__':
    backbone = resnet50(
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )
    overwrite_eps(backbone, 0.0)

    fpn = FeaturePyramidNetwork(
        input_keys_list=['res2', 'res3', 'res4', 'res5'],
        input_channels_list=[256, 512, 1024, 2048],
        output_channels=256,
        norm_layer=None,
        pool=False
    )

    model = FasterRCNNwithFPN(
        backbone,
        fpn,
        num_classes=91
    )
    model.load_state_dict(torch.load('data/faster_params.pth'))
    model.eval()

    img = Image.open('bb44.png')
    img_pth = torch.from_numpy(np.array(img))
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    img_pth = normalize_img(img_pth, image_mean, image_std)
    categories = _COCO_CATEGORIES
    colormap = generate_colormap(len(categories))
    boxes, scores, labels = model(img_pth)
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.95:
            x1, y1, x2, y2 = map(int, box)
            color = tuple(c for c in colormap[label])
            label_text = f"{categories[label]}: {score:.1%}"

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Text
            font = ImageFont.load_default(size=24)
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x1 + 5, y1 + 2), label_text, fill=color, font=font)

    img.show()
    img.save("detection_bb44.png")

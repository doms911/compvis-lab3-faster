import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision
from resnet import resnet50
from utils import normalize_img
from torchvision.models.detection._utils import overwrite_eps


if __name__ == '__main__':
    model = resnet50(
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )
    model.load_state_dict(torch.load('data/backbone_params.pth'))
    model = model.eval()
    overwrite_eps(model, 0.0)

    img = Image.open('bb44.png')
    img_pth = torch.from_numpy(np.array(img))
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])
    img_pth = normalize_img(img_pth, image_mean, image_std)

    print("Testing image normalization...")
    inputs_target = np.load("data/normalized_input_bb44.npy")

    if img_pth.shape != (3, 640, 960):
        print("Image shape is incorrect!")
        print("Check the permutation inside the normalization function!")
        exit()
    else:
        print("Image shape is correct!")
        if np.allclose(img_pth.numpy(), inputs_target):
            print("Normalized image values are correct!")
        else:
            print("Image values are not correct!")
            print("Check the normalization function!")

    print()
    out = model(img_pth.unsqueeze(0))
    print("Testing backbone outputs...")

    for key in ["res2", "res3", "res4", "res5"]:
        if key not in out:
            print(f"Features {key} are missing!")
            continue
        x = out[key].detach().numpy()
        x_target = np.load(f"data/{key}_bb44.npy")
        if x.shape == x_target.shape:
            print(f"Features {key} shape test: passed!")
            if np.allclose(x, x_target, atol=1e-4):
                print(f"Features {key} values test: passed!")
            else:
                print(f"Features {key} values test: failed!")
        else:
            print(f"Features {key} shape test: failed!")
        print("---------------------------------")

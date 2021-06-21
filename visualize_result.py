
import glob
import os

import cv2
#
# import numpy as np
import torch
import tqdm
from torch.backends import cudnn

from fastreid.utils.file_io import PathManager

cudnn.benchmark = True

from model.model import buildModel
import torch.nn.functional as F

def preprocess_image(batched_inputs):
    """
    Normalize and batch the input images.
    """
    if isinstance(batched_inputs, dict):
        images = batched_inputs['images']
    elif isinstance(batched_inputs, torch.Tensor):
        images = batched_inputs
    else:
        raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255]
    pixel_std=[0.229*255, 0.224*255, 0.225*255]
    pixel_mean=torch.Tensor(pixel_mean).view(1, -1, 1, 1).to(images.device)
    pixel_std = torch.Tensor(pixel_std).view(1, -1, 1, 1).to(images.device)
    images.sub_(pixel_mean).div_(pixel_std)
    return images


if __name__ == '__main__':
    img_size = [256, 192]
    SIZE_TEST=img_size

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feat_dim = 512
    num_classes = 15
    model = buildModel(feat_dim, num_classes).to(DEVICE)
    model.eval()

    modelfile="runs/Epoch-29.pth"

    state_dict=torch.load(modelfile, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)  # load


    classes_dict = {}
    with open("classes.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[idx] = class_name

    output="output/"
    PathManager.mkdirs(output)
    input_path = ["test/*.jpg"]
    save_path = output
    input = glob.glob(os.path.expanduser(input_path[0]))
    for path in tqdm.tqdm(input):
        img = cv2.imread(path)
        original_image=img
        # feat = demo.run_on_image(img)

        # the model expects RGB inputs
        original_image = original_image[:, :, ::-1]
        # Apply pre-processing to image.
        image = cv2.resize(original_image, tuple(SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for
        # network input
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        inputs = {"images": image.to(DEVICE)}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            images = preprocess_image(inputs)
            predictions = model(images)
            # Normalize feature to compute cosine distance
            features = F.normalize(predictions)
            features = features.cpu().data


        feat = features.numpy()
        class_name = classes_dict[feat.argmax()]
        imgname, _ = os.path.splitext(os.path.basename(path))
        save_img_name = f"{imgname}_{class_name}.jpg"
        cv2.imwrite(os.path.join(save_path, save_img_name), img)




# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

import numpy as np
from scipy.io import loadmat

from .bases import Dataset
from pathlib import Path
import glob
import os
from tqdm import tqdm
from PIL import Image,ExifTags
import logging
import torch

logger = logging.getLogger(__name__)

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

class CarAttr(Dataset):
    """Car attribute dataset.
    80k training images + 20k test images.
    The folder structure should be:
        pa100k/
            data/ # images
            annotation.mat
    """
    dataset_dir = 'output'

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)



        class_name = osp.join(self.dataset_dir, "classes.names")
        with open(class_name, 'r') as f:
            names = np.array([x for x in f.read().strip().splitlines()], dtype=np.str)  # labels
        attr_dict = {i: str(attr) for i, attr in enumerate(names)}
        self.num_classes = len(names)

        self.data_dir = "../data/gongjiaotrainmin.txt"
        train = self.extract_data()

        self.data_dir = "../data/gongjiaovalid.txt"

        val = self.extract_data()

        self.data_dir = "../data/gongjiaovalid.txt"

        test = self.extract_data()

        super(CarAttr, self).__init__(train, val, test, attr_dict=attr_dict, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    if len(l):
                        assert l.shape[1] == self.num_classes, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, self.num_classes), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, self.num_classes), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logger.info(f"New cache created: {path}")
        return x


    def extract_data(self):
        required_files = [self.data_dir]
        self.check_before_run(required_files)

        path = self.data_dir
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (path, e))

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels

        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0, f'No labels found in {cache_path}. Can not train without labels.'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        n = len(shapes)  # number of images

        data = []
        for i in range(n):
            attrs = self.labels[i][0]
            img_path = self.img_files[i]
            data.append((img_path, attrs))

        return data

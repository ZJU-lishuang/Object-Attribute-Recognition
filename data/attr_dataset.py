# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image,ExifTags
import os
import glob
from pathlib import Path
import logging
from torch._six import container_abcs, string_classes, int_classes

from fastreid.data.data_utils import read_image

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

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, np.ndarray):
        assert len(pic.shape) in (2, 3)
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def create_dataloader(root,img_size,batch_size,rank=-1,world_size=1, workers=8, image_weights=False):
    dataset=LoadImagesAndLabels(root,img_size)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    #build_reid_train_loader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=fast_batch_collator)
    return dataloader, dataset

def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class LoadImagesAndLabels(Dataset):
    def __init__(self,root='',img_size=[256,192],mode='train'):
        self.root = root
        self.img_size = img_size
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        class_name = osp.join(self.dataset_dir, "classes.names")
        with open(class_name, 'r') as f:
            names = np.array([x for x in f.read().strip().splitlines()], dtype=np.str)  # labels
        attr_dict = {i: str(attr) for i, attr in enumerate(names)}
        self.num_classes = len(names)

        self.data_dir = "../data/gongjiaotrain.txt"
        train = self.extract_data()

        self.data_dir = "../data/gongjiaovalid.txt"

        val = self.extract_data()

        self.data_dir = "../data/gongjiaovalid.txt"

        test = self.extract_data()

        self.train = train
        self.val = val
        self.test = test
        self._attr_dict = attr_dict
        self._num_attrs = len(self.attr_dict)

        if mode == 'train':
            self.data = self.train
        elif mode == 'val':
            self.data = self.val
        else:
            self.data = self.test

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

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        img_path, labels = self.img_items[index]
        img = read_image(img_path)

        # if self.transform is not None: img = self.transform(img)
        img = img.resize(self.img_size[0] if len(self.img_size) == 1 else self.img_size, interpolation=3)

        to_tensor(img)

        labels = torch.as_tensor(labels)

        return {
            "images": img,
            "targets": labels,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.attr_dict)

    @property
    def sample_weights(self):
        sample_weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, attr in self.img_items:
            sample_weights += torch.as_tensor(attr)
        sample_weights /= len(self)
        return sample_weights

class AttrDataset(Dataset):
    """Image Person Attribute Dataset"""

    def __init__(self, img_items, transform, attr_dict):
        self.img_items = img_items
        self.transform = transform
        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, labels = self.img_items[index]
        img = read_image(img_path)

        if self.transform is not None: img = self.transform(img)

        labels = torch.as_tensor(labels)

        return {
            "images": img,
            "targets": labels,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.attr_dict)

    @property
    def sample_weights(self):
        sample_weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, attr in self.img_items:
            sample_weights += torch.as_tensor(attr)
        sample_weights /= len(self)
        return sample_weights

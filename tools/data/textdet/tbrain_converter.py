import argparse
import glob
import json
import os.path as osp

from shutil import move

import mmcv
import numpy as np
from shapely.geometry import Polygon

from mmocr.utils import (convert_annotations, drop_orientation, is_not_png)


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory

    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    imgs_list = [
        drop_orientation(f) if is_not_png(f) else f for f in imgs_list
    ]

    files = []
    for img_file in imgs_list:
        gt_file = gt_dir + '/' + osp.splitext(osp.basename(img_file))[0] + '.json'
        files.append((img_file, gt_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        dataset(str): The dataset name, icdar2015 or icdar2017
        nproc(int): The number of process to collect annotations

    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')
    # read imgs with orientations as dataloader does when training and testing
    img_color = mmcv.imread(img_file, 'color')
    # make sure imgs have no orientations info, or annotation gt is wrong.
    assert img.shape[0:2] == img_color.shape[0:2]

    with open(gt_file) as f:
        raw_gt = json.load(f)
        gt_shapes = raw_gt['shapes']

    anno_info = []
    for gt_shape in gt_shapes:
        category_id = 1
        xy = gt_shape['points']
        coordinates = np.array(xy)
        polygon = Polygon(coordinates)
        iscrowd = 0
        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[xy])
        anno_info.append(anno)
    split_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.join(split_name, osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        anno_info=anno_info,
        segm_file=osp.join(split_name, osp.basename(gt_file)))
    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert tbrain annotations to COCO format'
    )
    parser.add_argument('tbrain_path', help='tbrain root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--split-list', nargs='+', help='a list of splits. e.g., "--split-list training test"')
    parser.add_argument('--split-from-training', action='store_true', default=False,
                        help='split validation set from training')
    parser.add_argument('--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tbrain_path = args.tbrain_path
    out_dir = args.out_dir if args.out_dir else tbrain_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(tbrain_path, 'imgs')
    gt_dir = osp.join(tbrain_path, 'annotations')

    if args.split_from_training:
        indexes = np.random.permutation(4000)
        print('Copying files from training set to validation set')
        for idx in indexes[:1000]:
            img_file_src = osp.join(img_dir, 'training', f'img_{idx}.jpg')
            img_file_dst = osp.join(img_dir, 'validation', f'img_{idx}.jpg')
            anno_file_src = osp.join(gt_dir, 'training', f'img_{idx}.json')
            anno_file_dst = osp.join(gt_dir, 'validation', f'img_{idx}.json')

            move(img_file_src, img_file_dst)
            move(anno_file_src, anno_file_dst)

    set_name = {}
    for split in args.split_list:
        set_name.update({split: 'instances_' + split + '.json'})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(print_tmpl='It takes {}s to convert icdar annotation'):
            files = collect_files(osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            convert_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()

import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import init_detector
from tqdm import tqdm

from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = ArgumentParser()
    parser.add_argument('img_folder', help='Image folder.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('out_folder', help='Path to save visualized image.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline

    submission = []
    image_folder = Path(args.img_folder)
    image_files = [image_file for image_file in image_folder.glob('*')]
    for image_file in tqdm(image_files):
        # test a single image
        result = model_inference(model, str(image_file))

        for boundary_result in result['boundary_result']:
            pred = list(map(lambda x: str(int(x)), boundary_result[:-1]))
            pred_str = ','.join(pred)
            submission.append(f'{image_file.stem[4:]},{pred_str},{boundary_result[-1]}\n')

        # show the results
        out_file = osp.join(args.out_folder, f'result_{image_file.name}')
        img = model.show_result(image_file, result, out_file=out_file, show=False)

        if img is None:
            img = mmcv.imread(image_file)

        mmcv.imwrite(img, out_file)

    print('Outputting submission.csv...')
    submission = ''.join(submission)
    with open('submission.csv', 'w') as f:
        f.write(submission)

    print('Done.')


if __name__ == '__main__':
    main()

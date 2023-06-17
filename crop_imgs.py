import logging
from inference import resize_cut
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
from auxiliary import logger

DIRS = ['Images', 'Labels', 'TrainId']

def process(indir, outdir, resolution, extension):
    for p in tqdm(list(indir.glob(f'*.{args.extension}'))):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        tiles = resize_cut(img, resolution)

        for i in range(len(tiles)):
            for j in range(len(tiles[i])):
                outpath = outdir / f'{p.stem}_{i}{j}.{extension}'

                cv2.imwrite(str(outpath), tiles[i][j])

def main(args):
    if args.debug:
        logger.setLevel(logging.DEBUG)

    args.outdir.mkdir(exist_ok=True)

    res = args.resolution.split('x')
    resolution = (int(res[1]), int(res[0]))

    for seq in args.indir.glob('seq*'):
        for folder in DIRS:
            if (seq / folder).exists():
                logger.info(f'Processing sequence {seq.name}, folder {folder}')
                outdir = args.outdir / seq.name / folder
                outdir.mkdir(exist_ok=True, parents=True)
                logger.debug(f'Path {seq / folder}')
                process(seq / folder, outdir, resolution, args.extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='indir', type=Path, required=True)
    parser.add_argument('-o', dest='outdir', type=Path, required=True)
    parser.add_argument('-e', dest='extension', type=Path, default='png')
    parser.add_argument('-r', dest='resolution', type=str)
    parser.add_argument('-d', dest='debug', action='store_true')

    args = parser.parse_args()
    main(args)

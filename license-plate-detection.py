#!/usr/bin/env python3
"""License plate detection using WPOD-Net."""
import sys
import cv2
import traceback

from glob import glob
from os.path import splitext, basename

from src.keras_utils import load_model, detect_lp
from src.utils import im2single
from src.label import Shape, writeShapes


def main():
    try:
        if len(sys.argv) < 3:
            print('用法: python license-plate-detection.py <input_dir> <lp_model_path>')
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = input_dir
        lp_threshold = 0.5
        wpod_net_path = sys.argv[2]
        wpod_net = load_model(wpod_net_path)

        imgs_paths = glob('%s/*car.png' % input_dir)
        print('Searching for license plates using WPOD-NET')

        for i, img_path in enumerate(imgs_paths):
            print('\t Processing %s' % img_path)
            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2**4)), 608)
            print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

            Llp, LlpImgs, _ = detect_lp(
                wpod_net, im2single(Ivehicle),
                bound_dim, 2**4, (240, 80), lp_threshold
            )

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)

                cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
                writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()

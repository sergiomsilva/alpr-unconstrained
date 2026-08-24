#!/usr/bin/env python3
"""Generate annotated output images and CSV results."""
import sys

from glob import glob
from os.path import splitext, basename, isfile

from src.utils import image_files_from_folder
from src.drawing_utils import draw_label, draw_losangle, write2img
from src.label import lread, Label, readShapes


YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def image_files(folder):
    files = []
    for ext in ('jpg', 'jpeg', 'png'):
        files += glob('%s/*.%s' % (folder, ext))
        files += glob('%s/*.%s' % (folder, ext.upper()))
    return sorted(files)


def main():
    if len(sys.argv) < 3:
        print('用法: python gen-outputs.py <input_dir> <output_dir>')
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    img_files = image_files(input_dir)

    for img_file in img_files:
        bname = splitext(basename(img_file))[0]
        I = cv2_read(img_file)
        if I is None:
            continue

        detected_cars_labels = '%s/%s_cars.txt' % (output_dir, bname)
        Lcar = lread(detected_cars_labels)

        sys.stdout.write('%s' % bname)

        if Lcar:
            for i, lcar in enumerate(Lcar):
                draw_label(I, lcar, color=YELLOW, thickness=3)

                lp_label = '%s/%s_%dcar_lp.txt' % (output_dir, bname, i)
                lp_label_str = '%s/%s_%dcar_lp_str.txt' % (output_dir, bname, i)

                if isfile(lp_label):
                    Llp_shapes = readShapes(lp_label)
                    pts = Llp_shapes[0].pts * lcar.wh().reshape(2, 1) + lcar.tl().reshape(2, 1)
                    ptspx = pts * _img_wh(I).reshape(2, 1)
                    draw_losangle(I, ptspx, RED, 3)

                    if isfile(lp_label_str):
                        with open(lp_label_str, 'r') as f:
                            lp_str = f.read().strip()
                        llp = Label(0, tl=pts.min(1), br=pts.max(1))
                        write2img(I, llp, lp_str)
                        sys.stdout.write(',%s' % lp_str)

        cv2_write('%s/%s_output.png' % (output_dir, bname), I)
        sys.stdout.write('\n')


def cv2_read(path):
    import cv2
    return cv2.imread(path)


def cv2_write(path, img):
    import cv2
    cv2.imwrite(path, img)


def _img_wh(img):
    import numpy as np
    return np.array(img.shape[1::-1], dtype=float)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""License plate OCR using PaddleOCR."""
import sys
import traceback
from os.path import splitext, basename
from glob import glob
from paddleocr import PaddleOCR


def main():
    try:
        if len(sys.argv) < 2:
            print('用法: python license-plate-ocr.py <input_dir>')
            sys.exit(1)

        input_dir = sys.argv[1]
        output_dir = input_dir

        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

        print('Performing OCR with PaddleOCR...')
        for img_path in imgs_paths:
            print('\tScanning %s' % img_path)
            bname = basename(splitext(img_path)[0])

            result = ocr.ocr(img_path, cls=True)
            if result and result[0]:
                lines = result[0]
                lines.sort(key=lambda x: x[0][0][0])
                text = ''.join([line[1][0] for line in lines])
                with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
                    f.write(text + '\n')
                print('\t\tLP: %s' % text)
            else:
                print('\t\tNo characters found')

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()

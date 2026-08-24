#!/usr/bin/env python3
"""Vehicle detection using YOLOv8."""
import sys
import cv2
import numpy as np
import traceback
from os.path import splitext, basename, isdir
from os import makedirs
from glob import glob
from ultralytics import YOLO
from src.label import Label, lwrite
from src.utils import crop_region


def image_files_from_folder(folder, extensions=('jpg', 'jpeg', 'png')):
    files = []
    for ext in extensions:
        files += glob('%s/*.%s' % (folder, ext))
        files += glob('%s/*.%s' % (folder, ext.upper()))
    return sorted(files)


def detect_vehicles(image_path, output_dir, model, vehicle_threshold=0.5,
                    vehicle_classes=(2, 5, 7)):
    """对单张图片做车辆检测，返回 Lcars 和保存裁剪图。"""
    bname = splitext(basename(image_path))[0]
    results = model(image_path, conf=vehicle_threshold, classes=vehicle_classes, verbose=False)

    Iorig = cv2.imread(image_path)
    if Iorig is None:
        return [], []
    WH = np.array(Iorig.shape[1::-1], dtype=float)
    Lcars = []
    rects = []

    if len(results) and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tl = np.array([x1 / WH[0], y1 / WH[1]])
            br = np.array([x2 / WH[0], y2 / WH[1]])
            label = Label(0, tl, br)
            Icar = crop_region(Iorig, label)
            if Icar is not None:
                cv2.imwrite('%s/%s_%dcar.png' % (output_dir, bname, i), Icar)
                Lcars.append(label)
                rects.append([x1, y1, x2, y2])

        lwrite('%s/%s_cars.txt' % (output_dir, bname), Lcars)

    return Lcars, rects


def main():
    """CLI 入口"""
    try:
        if len(sys.argv) < 3:
            print('用法: python vehicle-detection.py <input_dir> <output_dir>')
            sys.exit(1)
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        vehicle_threshold = 0.5
        VEHICLE_CLASSES = [2, 5, 7]

        model = YOLO('yolov8n.pt')

        imgs_paths = image_files_from_folder(input_dir)
        if not isdir(output_dir):
            makedirs(output_dir)

        print('Searching for vehicles using YOLOv8...')
        for img_path in imgs_paths:
            print('\tScanning %s' % img_path)
            Lcars, _ = detect_vehicles(img_path, output_dir, model,
                                       vehicle_threshold, VEHICLE_CLASSES)
            print('\t\t%d vehicles found' % len(Lcars))

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ALPR Web UI — 车牌识别图形化界面
基于 Flask 的轻量级 Web 应用，支持图片上传和一键识别。
"""
import os
import cv2
import numpy as np
import time
import uuid
import logging

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.keras_utils import load_model, detect_lp
from src.utils import im2single
from src.label import Label, Shape, lwrite, writeShapes, readShapes
from src.drawing_utils import draw_label, draw_losangle, write2img
from src.utils import crop_region

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'web_output')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LP_THRESHOLD = 0.5
VEHICLE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 5, 7]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_models = {}

def get_models():
    if not _models:
        log.info("Loading YOLOv8 vehicle detector...")
        _models['yolo'] = YOLO('yolov8n.pt')
        log.info("Loading WPOD-Net license plate detector...")
        wpod_path = 'data/lp-detector/wpod-net_update1.h5'
        if os.path.exists(wpod_path):
            _models['wpod'] = load_model(wpod_path)
        else:
            log.warning("WPOD-Net not found at %s. Run: bash get-networks.sh", wpod_path)
            _models['wpod'] = None
        log.info("Loading PaddleOCR...")
        _models['ocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        log.info("All models loaded.")
    return _models

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, session_id):
    outdir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(outdir, exist_ok=True)
    models = get_models()
    timing = {}
    vehicles = []
    plates = []

    Iorig = cv2.imread(image_path)
    if Iorig is None:
        raise ValueError("Cannot read image: %s" % image_path)
    WH = np.array(Iorig.shape[1::-1], dtype=float)

    t0 = time.time()
    results = models['yolo'](image_path, conf=VEHICLE_THRESHOLD, classes=VEHICLE_CLASSES, verbose=False)
    timing['vehicle_detection'] = time.time() - t0
    bname = os.path.splitext(os.path.basename(image_path))[0]
    Lcars = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tl = np.array([x1 / WH[0], y1 / WH[1]])
            br = np.array([x2 / WH[0], y2 / WH[1]])
            label = Label(0, tl, br)
            Icar = crop_region(Iorig, label)
            if Icar is not None:
                cv2.imwrite(os.path.join(outdir, '%s_%dcar.png' % (bname, i)), Icar)
                Lcars.append(label)
                vehicles.append({'index': i})
        lwrite(os.path.join(outdir, '%s_cars.txt' % bname), Lcars)

    t0 = time.time()
    if models['wpod'] is not None and Lcars:
        for i, lcar in enumerate(Lcars):
            Ivehicle = cv2.imread(os.path.join(outdir, '%s_%dcar.png' % (bname, i)))
            if Ivehicle is None: continue
            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            bound_dim = min(int(ratio * 288.) + (int(ratio * 288.) % (2**4)), 608)
            Llp, LlpImgs, _ = detect_lp(models['wpod'], im2single(Ivehicle), bound_dim, 2**4, (240, 80), LP_THRESHOLD)
            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(outdir, '%s_%dlp.png' % (bname, i)), Ilp * 255.)
                writeShapes(os.path.join(outdir, '%s_%dlp.txt' % (bname, i)), [Shape(Llp[0].pts)])
                plates.append({'vehicle_index': i, 'lp_image': '%s_%dlp.png' % (bname, i)})
    timing['lp_detection'] = time.time() - t0

    t0 = time.time()
    for p in plates:
        r = models['ocr'].ocr(os.path.join(outdir, p['lp_image']), cls=True)
        if r and r[0]:
            lines = sorted(r[0], key=lambda x: x[0][0][0])
            p['text'] = ''.join([l[1][0] for l in lines])
            p['confidence'] = max([l[1][1] for l in lines])
            with open(os.path.join(outdir, '%s_str.txt' % os.path.splitext(p['lp_image'])[0]), 'w') as f:
                f.write(p['text'] + '\n')
        else:
            p['text'], p['confidence'] = '', 0
    timing['ocr'] = time.time() - t0

    t0 = time.time()
    annotated = Iorig.copy()
    for i, lcar in enumerate(Lcars):
        draw_label(annotated, lcar, color=(0,255,255), thickness=3)
        lp_path = os.path.join(outdir, '%s_%dlp.txt' % (bname, i))
        sp_path = os.path.join(outdir, '%s_str.txt' % ('%s_%d' % (bname, i)))
        if os.path.exists(lp_path):
            shapes = readShapes(lp_path)
            if shapes:
                pts = shapes[0].pts * lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                draw_losangle(annotated, pts * WH.reshape(2,1), (0,0,255), 3)
                if os.path.exists(sp_path):
                    with open(sp_path) as f:
                        write2img(annotated, Label(0, pts.min(1), pts.max(1)), f.read().strip())
    cv2.imwrite(os.path.join(outdir, 'annotated.jpg'), annotated)
    timing['annotation'] = time.time() - t0
    timing['total'] = sum(timing.values())
    return {'session_id': session_id, 'vehicle_count': len(vehicles), 'plates': plates,
            'timing': {k: '%.2fs' % v for k, v in timing.items()}, 'annotated_image': 'annotated.jpg',
            'uploaded_filename': os.path.basename(image_path)}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='请选择一张图片')
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', error='请选择一张有效的 JPG/PNG 图片')
        try:
            sid = uuid.uuid4().hex[:12]
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename('%s_%s' % (sid, file.filename)))
            file.save(path)
            result = process_image(path, sid)
            return render_template('index.html', result=result, uploaded_filename=result['uploaded_filename'])
        except Exception as e:
            log.exception("Processing failed")
            return render_template('index.html', error='处理失败: %s' % str(e))
    return render_template('index.html', result=None)

@app.route('/output/<session_id>/<filename>')
def serve_output(session_id, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], session_id), filename)

@app.route('/upload/<filename>')
def serve_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

def main():
    log.info("Starting ALPR Web UI at http://127.0.0.1:5000")
    log.info("Download models first: bash get-networks.sh")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()

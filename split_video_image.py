import cv2
import os
import shutil

def split_video(in_video, out_dir):
	vidcap = cv2.VideoCapture(in_video)
	success, image = vidcap.read()
	count = 0

	while success:
		cv2.imwrite("%s/v%09d.jpg" % (out_dir, count), image)
		success, image = vidcap.read()
		count += 1

def combine_video(in_dir, out_video):
	images = [f for f in os.listdir(in_dir)]
	img = cv2.imread("%s/%s" % (in_dir, images[0]))
	size = (img.shape[1], img.shape[0])
	out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

	for image in images:
		img = cv2.imread("%s/%s" % (in_dir, image))
		out.write(img)

	out.release()

in_dir = "tmp_in"
out_dir = "tmp_out"

in_video = "samples/sample1_short.mp4"
out_video = "samples/sample1_short_tagged.mp4"
lp_model="data/lp-detector/wpod-net_update1.h5"
csv_file="samples/results.csv"

if os.path.exists(in_dir):
	shutil.rmtree(in_dir)
	shutil.rmtree(out_dir)
os.mkdir(in_dir)
os.mkdir(out_dir)

split_video(in_video, in_dir)

os.system("python vehicle-detection.py %s %s" % (in_dir, out_dir))
os.system("python license-plate-detection.py %s %s" % (out_dir, lp_model))
os.system("python license-plate-ocr.py %s" % (out_dir))
os.system("python gen-outputs.py %s %s > %s" % (in_dir, out_dir, csv_file))

for out_file in os.listdir(out_dir):
    if not out_file.endswith("output.png"):
        os.remove("%s/%s" % (out_dir, out_file))

combine_video(out_dir, out_video)
#shutil.rmtree(in_dir)
#shutil.rmtree(out_dir)

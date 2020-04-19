import cv2
import os
import shutil
import sys
import time

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
	images.sort()
	
	for image in images:
		img = cv2.imread("%s/%s" % (in_dir, image))
		out.write(img)

	out.release()

if len(sys.argv) < 2:
	print("Pls provide input video name (with extension)")
	quit()

start_time = time.time()
vid_name = sys.argv[1]
in_dir = "tmp_in"
out_dir = "tmp_out"

in_video = vid_name
out_video = "%s_tagged.mp4" % (vid_name)
lp_model="data/lp-detector/wpod-net_update1.h5"

if os.path.exists(in_dir):
	shutil.rmtree(in_dir)
	shutil.rmtree(out_dir)
os.mkdir(in_dir)
os.mkdir(out_dir)

split_video(in_video, in_dir)

os.system("python vehicle-detection.py %s %s" % (in_dir, out_dir))
os.system("python license-plate-detection.py %s %s" % (out_dir, lp_model))
os.system("python license-plate-ocr.py %s" % (out_dir))
os.system("python gen-outputs.py %s %s" % (in_dir, out_dir))

for out_file in os.listdir(out_dir):
    if not out_file.endswith("output.png"):
        os.remove("%s/%s" % (out_dir, out_file))

combine_video(out_dir, out_video)
#shutil.rmtree(in_dir)
#shutil.rmtree(out_dir)

print("Execution time: %fs" % (time.time() - start_time))

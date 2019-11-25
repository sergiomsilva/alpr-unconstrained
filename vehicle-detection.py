import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect
from time import sleep
import datetime

if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]

		vehicle_threshold = .5

		# vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		# vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		# vehicle_dataset = 'data/vehicle-detector/voc.data'
		vehicle_weights = 'data/vehicle-detector/yolov2.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolov2.cfg'
		vehicle_dataset = 'data/vehicle-detector/coco.data'

		vehicle_net  = dn.load_net(vehicle_netcfg.encode('utf-8'), vehicle_weights.encode('utf-8'), 0)
		vehicle_meta = dn.load_meta(vehicle_dataset.encode('utf-8'))

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print ('Searching for vehicles using YOLO...')

		for i,img_path in enumerate(imgs_paths):
			start = datetime.datetime.now()
			print ('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0].encode('utf-8'))
			# img = cv2.imread(img_path)
			img = img_path.encode('utf-8')
			print(img)

			R,_ = detect(vehicle_net, vehicle_meta, img ,thresh=vehicle_threshold)
			# print(detect(vehicle_net, vehicle_meta, img ,thresh=vehicle_threshold))
			# print("R :",R)
			R = [r for r in R if r[0].decode('utf-8') in ['car','bus']]

			print ('\t\t%d cars found' % len(R))

			if len(R):

				Iorig = cv2.imread(img_path)
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				Lcars = []

				for i,r in enumerate(R):

					cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
					tl = np.array([cx - w/2., cy - h/2.])
					br = np.array([cx + w/2., cy + h/2.])
					label = Label(0,tl,br)
					Icar = crop_region(Iorig,label)

					Lcars.append(label)

					cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

				lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)
			stop = datetime.datetime.now()
			print(stop-start)

	except:
		traceback.print_exc()
		sys.exit(1)
	sleep(20)
	sys.exit(0)
	
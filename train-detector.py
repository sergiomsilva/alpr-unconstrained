
import sys
import numpy as np
import cv2
import argparse

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.detector_network import create_model
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-m' 		,'--model'			,type=str 					,help='Path to previous model')
	parser.add_argument('-n' 		,'--name'			,type=str , required=True	,help='Model name')
	parser.add_argument('-id'		,'--input-dir'		,type=str , required=True	,help='Input data directories for training')
	parser.add_argument('-its'		,'--iterations'		,type=int , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int , default=32		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--outdir'			,type=str , default='./'	,help='Output directory (default = ./)')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.input_dir
	modelpath 	= args.model
	outdir 		= args.outdir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	if modelpath:
		model = load_model(modelpath)
	else:
		model = create_model()

	model_stride = 2**4

	model.compile(loss=loss, optimizer='adam')

	print 'Checking input directory...'
	Files = image_files_from_folder(train_dir)

	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			I = cv2.imread(file)
			Data.append([I,L[0]])

	print '%d images+labels found' % len(Data)


	Xtrain = np.empty((batch_size,dim,dim,3),dtype='single')
	Ytrain = np.empty((batch_size,dim/model_stride,dim/model_stride,2*4+1))

	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)

	for it in range(iterations):

		print 'Iter. %d (of %d)' % (it+1,iterations)

		for k in range(batch_size):
			data = choice(Data)
			XX,llp,pts = augment_sample(data[0],data[1].pts,dim)
			YY = labels2output_map(llp,pts,dim,model_stride)

			Xtrain[k] = XX
			Ytrain[k] = YY

		train_loss = model.train_on_batch(Xtrain,Ytrain)

		print '\tLoss: %f' % train_loss

		# Save model every 1000 iterations
		if (it+1) % 1000 == 0:
			print 'Saving model (%s)' % model_path_backup
			save_model(model,model_path_backup)

	print 'Saving model (%s)' % model_path_final
	save_model(model,model_path_final)

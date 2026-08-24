#!/usr/bin/env python3
"""Train WPOD-Net license plate detector."""
import sys
import numpy as np
import argparse
import cv2

from os.path import isfile, isdir, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator


def load_network(modelpath, input_dim):
    model = load_model(modelpath)
    import keras
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)
    # Keras 3: shape dims are plain ints / None, no .value attribute anymore
    output_shape = tuple(int(s) if s is not None else None for s in outputs.shape[1:])
    output_dim = output_shape[1]
    model_stride = input_dim / output_dim
    assert input_dim % output_dim == 0, 'The output resolution must be divisible by the input resolution'
    assert model_stride == 2**4, 'Make sure your model generates a feature map with resolution 16x smaller than the input'
    return model, model_stride, None, output_shape


def process_data_item(data_item, dim, model_stride):
    XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim)
    YY = labels2output_map(llp, pts, dim, model_stride)
    return XX, YY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to previous model')
    parser.add_argument('-n', '--name', type=str, required=True, help='Model name')
    parser.add_argument('-tr', '--train-dir', type=str, required=True, help='Input data directory for training')
    parser.add_argument('-its', '--iterations', type=int, default=300000, help='Number of mini-batch iterations (default = 300000)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='Mini-batch size (default = 32)')
    parser.add_argument('-od', '--output-dir', type=str, default='./', help='Output directory (default = ./)')
    parser.add_argument('-op', '--optimizer', type=str, default='Adam', help='Optimizer (default = Adam)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=.01, help='Optmizer LR (default = 0.01)')
    args = parser.parse_args()

    netname = args.name
    train_dir = args.train_dir
    outdir = args.output_dir
    iterations = args.iterations
    batch_size = args.batch_size
    dim = 208

    if not isdir(outdir):
        makedirs(outdir)

    model, model_stride, xshape, yshape = load_network(args.model, dim)

    import keras
    opt = getattr(keras.optimizers, args.optimizer)(learning_rate=args.learning_rate)
    model.compile(loss=loss, optimizer=opt)

    print('Checking input directory...')
    Files = image_files_from_folder(train_dir)

    Data = []
    for file in Files:
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            L = readShapes(labfile)
            I = cv2.imread(file)
            Data.append([I, L[0]])

    print('%d images with labels found' % len(Data))

    from os.path import basename
    dg = DataGenerator(
        data=Data,
        process_data_item_func=lambda x: process_data_item(x, dim, model_stride),
        xshape=(dim, dim, 3),
        yshape=(dim // model_stride, dim // model_stride, yshape[2] + 1),
        nthreads=2,
        pool_size=1000,
        min_nsamples=100
    )
    dg.start()

    Xtrain = np.empty((batch_size, dim, dim, 3), dtype='float32')
    Ytrain = np.empty((batch_size, int(dim // model_stride), int(dim // model_stride), 2 * 4 + 1))

    model_path_backup = '%s/%s_backup' % (outdir, netname)
    model_path_final = '%s/%s_final' % (outdir, netname)

    for it in range(iterations):
        print('Iter. %d (of %d)' % (it + 1, iterations))
        Xtrain, Ytrain = dg.get_batch(batch_size)
        train_loss = model.train_on_batch(Xtrain, Ytrain)
        print('\tLoss: %f' % train_loss)
        if (it + 1) % 1000 == 0:
            print('Saving model (%s)' % model_path_backup)
            save_model(model, model_path_backup)

    print('Stopping data generator')
    dg.stop()
    print('Saving model (%s)' % model_path_final)
    save_model(model, model_path_final)


if __name__ == '__main__':
    main()

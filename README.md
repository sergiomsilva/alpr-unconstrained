# ALPR in Unscontrained Scenarios

## Introduction

This repository contains the author's implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

## Requirements

In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:

```shellscript
$ cd darknet && make
```

The current version was tested in an Ubuntu 16.04 machine, with Keras 2.0.6 and TensorFlow 1.5.0.

## Download Models

After building the Darknet framework, you must execute the "get-networks.sh" script. This will download all the trained models:

```shellscript
$ bash get-networks.sh
```

## Running a simple test

Use the script "run.sh" to run our ALPR approach. It requires 3 arguments:
* __Input directory:__ should contain at least 1 image in JPG or PNG format;
* __Output directory:__ during the recognition process, many temporary files will be generated inside this directory and erased in the end. The remaining files will be related to the automatic annotated image;
* __CSV file:__ specify an output CSV file.

```shellscript
$ bash run.sh samples/test /tmp/output /tmp/output/results.csv
```

## Training the LP detector

To train the LP detector network from scratch, or fine-tuning it for new samples, you can use the train-detector.py script. In folder samples/train-detector there are 3 annotated samples which are used just for demonstration purposes. To correctly reproduce our experiments, this folder must be filled with all the annotations provided in the training set, and their respective images transferred from the original datasets.

The following command can be used to train the network from scratch considering the data inside the train-detector folder:

```shellscript
$ python train-detector.py --name new-network --outdir /tmp/ --input-dir samples/train-detector
```

For fine-tunning, add "-m data/lp-detector/wpod-net" to the command line above.

## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.

## Further information

* Paper webpage: http://www.inf.ufrgs.br/~smsilva/alpr-unconstrained/
* Datasets: http://www.inf.ufrgs.br/~crjung/alpr-datasets


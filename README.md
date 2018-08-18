# ALPR in Unscontrained Scenarios

## Introduction

This repository contains the author's implementation of ECCV 2018 paper "Automatic License Plate Recognition in Unconstrained Scenarios".

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
$ bash run.sh samples/ /tmp/output /tmp/output/results.csv
```

## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.

## Further information

* Paper webpage: http://www.inf.ufrgs.br/~smsilva/alpr-unconstrained/
* Dataset annotations used for training: http://www.inf.ufrgs.br/~crjung/alpr-datasets


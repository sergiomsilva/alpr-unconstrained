# ALPR in Unscontrained Scenarios

## Introduction

This repository contains the author's implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

* Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

* The original repo can be found [here] (https://github.com/sergiomsilva/alpr-unconstrained)

## Changes made

*The changes I have made to this repo are*
1. Converted the code from legacy *Python 2* to *Python 3* for the present and the future. An exception is the **annotation tool (annotation_tool.py)** which has to be used in a Python 2 environment
2. To annotate the training data, the *folder path* can be passed as argument instead of passing the individual files.
#!/bin/bash

set -e

mkdir -p data/lp-detector

# WPOD-Net license plate detector model (original paper weights)
wget -c -N http://sergiomsilva.com/data/eccv2018/lp-detector/wpod-net_update1.h5   -P data/lp-detector/
wget -c -N http://sergiomsilva.com/data/eccv2018/lp-detector/wpod-net_update1.json -P data/lp-detector/

echo ""
echo "WPOD-Net model downloaded."
echo "YOLOv8 and PaddleOCR models will be auto-downloaded on first use."

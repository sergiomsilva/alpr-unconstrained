#!/bin/bash
set -e

input_dir=''
output_dir=''
csv_file=''
lp_model='data/lp-detector/wpod-net_update1.h5'

usage() {
    echo ""
    echo " Usage:"
    echo ""
    echo "   bash $0 -i input/dir -o output/dir -c csv_file.csv [-h] [-l path/to/model]:"
    echo ""
    echo "   -i   Input dir path (containing JPG or PNG images)"
    echo "   -o   Output dir path"
    echo "   -c   Output CSV file path"
    echo "   -l   Path to Keras LP detector model (default = $lp_model)"
    echo "   -h   Print this help information"
    echo ""
    exit 1
}

while getopts 'i:o:c:l:h' OPTION; do
    case $OPTION in
        i) input_dir=$OPTARG;;
        o) output_dir=$OPTARG;;
        c) csv_file=$OPTARG;;
        l) lp_model=$OPTARG;;
        h) usage;;
    esac
done

if [ -z "$input_dir"  ]; then echo "Input dir not set."; usage; exit 1; fi
if [ -z "$output_dir" ]; then echo "Output dir not set."; usage; exit 1; fi
if [ -z "$csv_file"   ]; then echo "CSV file not set."; usage; exit 1; fi

if [ ! -d "$input_dir" ]; then
    echo "Input directory ($input_dir) does not exist"
    exit 1
fi

mkdir -p "$output_dir"

echo "=== Stage 1: Vehicle detection (YOLOv8) ==="
python3 vehicle-detection.py "$input_dir" "$output_dir"

echo "=== Stage 2: License plate detection (WPOD-Net) ==="
python3 license-plate-detection.py "$output_dir" "$lp_model"

echo "=== Stage 3: OCR (PaddleOCR) ==="
python3 license-plate-ocr.py "$output_dir"

echo "=== Stage 4: Generate output ==="
python3 gen-outputs.py "$input_dir" "$output_dir" > "$csv_file"

# Clean intermediate files
rm -f "$output_dir"/*_lp.png "$output_dir"/*car.png
rm -f "$output_dir"/*_cars.txt "$output_dir"/*_lp.txt "$output_dir"/*_str.txt

echo "Done! Results saved to $csv_file"

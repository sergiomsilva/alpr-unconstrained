#!/bin/bash

check_file() 
{
	if [ ! -f "$1" ]
	then
		return 0
	else
		return 1
	fi
}

check_dir() 
{
	if [ ! -d "$1" ]
	then
		return 0
	else
		return 1
	fi
}


# Check if Darknet is compiled
check_file "darknet/libdarknet.so"
retval=$?
if [ $retval -eq 0 ]
then
	echo "Darknet is not compiled! Go to 'darknet' directory and 'make'!"
	exit 0
fi

# Check # of arguments
if [ ! $# -eq 3 ]
then
	echo ""
	echo " Required arguments:"
	echo ""
	echo "   1. Input dir path (containing JPG or PNG images)"
	echo "   2. Output dir path"
	echo "   3. Output CSV file path"
	echo ""
	exit 1
fi

read -n1 -r -p "Press any key to continue..." key


# Download all networks
bash get-networks.sh


# Check if input dir exists
check_dir $1
retval=$?
if [ $retval -eq 0 ]
then
	echo "Input directory ($1) does not exist"
	exit 0
fi

# Check if output dir exists, if not, create it
check_dir $2
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $2
fi

# Detect vehicles
python vehicle-detection.py $1 $2

# Detect license plates
python license-plate-detection.py $2 data/lp-detector/wpod-net.h5

# OCR
python license-plate-ocr.py $2

# Draw output and generate list
python gen-outputs.py $1 $2 > $3

# Clean files and draw output
rm $2/*_lp.png
rm $2/*car.png
rm $2/*_cars.txt
rm $2/*_lp.txt
rm $2/*_str.txt
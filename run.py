import os

lp_model="data/lp-detector/wpod-net_update1.h5"
input_dir="samples/test"
output_dir="samples/out"
csv_file="samples/out/results.csv"

# Detect vehicles
os.system("python vehicle-detection.py %s %s)" % (input_dir, output_dir))

# Detect license plates
os.system("python license-plate-detection.py %s %s)" % (output_dir, lp_model))

# OCR
os.system("python license-plate-ocr.py %s)" % (output_dir))

# Draw output and generate list
os.system("python gen-outputs.py %s %s > %s)" % (input_dir, output_dir, csv_file))

# Clean files and draw output
os.remove("%s/*_lp.png" % (output_dir))
os.remove("%s/*car.png" % (output_dir))
os.remove("%s/*_cars.txt" % (output_dir))
os.remove("%s/*_lp.txt" % (output_dir))
os.remove("%s/*_str.txt" % (output_dir))

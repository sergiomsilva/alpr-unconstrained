pip3 install --upgrade tensorflow-gpu
pip3 install numpy scipy scikit-learn pillow h5py
pip3 install keras
mv ./yolov2.cfg ./data/vehicle-detector
mv ./coco.data ./data/vehicle-detector
mv ./coco.names ./data/vehicle-detector
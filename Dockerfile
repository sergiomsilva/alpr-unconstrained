FROM tensorflow/tensorflow:1.5.0-gpu
USER root

ARG NAME=alpr

WORKDIR /home/${NAME}

# Install project related software
RUN apt-get update && \
    apt-get install -y \
    wget \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev

# Install project related python packages
RUN pip install --upgrade pip && \
    pip install keras==2.2.4

# install opencv
RUN wget https://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.13/opencv-2.4.13.zip && \
    unzip opencv-2.4.13.zip && \
    cd opencv-2.4.13/ && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_PYTHON_SUPPORT=ON -D WITH_XINE=ON -D WITH_TBB=ON .. && \
    make -j$(nproc) && make install -j$(nproc) && \
    cd ../.. && \
    rm opencv-2.4.13.zip

COPY . .

RUN cd /home/${NAME}/darknet && \
    make && \
    cd ..

RUN bash get-networks.sh
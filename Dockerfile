FROM python:3.11-slim

RUN apt-get update \
        && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbbmalloc2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

WORKDIR /app
ENV OPENCV_VERSION="4.11.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
        && unzip ${OPENCV_VERSION}.zip \
        && mkdir ./opencv-${OPENCV_VERSION}/cmake_binary \
        && cd ./opencv-${OPENCV_VERSION}/cmake_binary \
        && cmake -DBUILD_TIFF=ON \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=ON \
        -DWITH_TBB=ON \
        -DWITH_EIGEN=OFF \
        -DWITH_V4L=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$(python3.11 -c "import sys; print(sys.prefix)") \
        -DPYTHON_EXECUTABLE=$(which python3.11) \
        -DPYTHON_INCLUDE_DIR=$(python3.11 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_PACKAGES_PATH=$(python3.11 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        .. \
        && make -j$(nproc) install \
        && rm /app/${OPENCV_VERSION}.zip \
        && rm -r /app/opencv-${OPENCV_VERSION}

RUN ln -s \
  /usr/local/python/cv2/python-3.11/cv2.cpython-37m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.11/site-packages/cv2.so

RUN apt-get --fix-missing update \
    && apt-get --fix-broken install \
    && apt-get install -y poppler-utils \
    && apt-get install -y tesseract-ocr \
    && apt-get install -y tesseract-ocr-deu \
    && apt-get install -y libtesseract-dev \
    && apt-get install -y libleptonica-dev \
    && ldconfig \
    && apt install -y libsm6 libxext6


COPY ./requirements.txt ./ 
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src ./src

EXPOSE 42069

CMD ["python", "src/server.py"]

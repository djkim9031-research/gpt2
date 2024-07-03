FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

# Set environment variables for locale 
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    git \
    g++-9 \
    gcc-9 \
    wget \
    curl \
    unzip \
    libopencv-dev \
    python3-dev \
    python3-pip &&   \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install numpy matplotlib

# Install CMake 3.29.6
RUN cd /opt && wget https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6.tar.gz && \
    tar -zxvf cmake-3.29.6.tar.gz && \
    cd cmake-3.29.6 && \
    ./bootstrap && \
    make -j${nproc} && make install && \
    rm ../cmake-3.29.6.tar.gz 

# Clone libtorch
RUN mkdir /opt/libtorch && \
    cd /opt/libtorch && \
    curl -L -o libtorch-gpu.zip https://download.pytorch.org/libtorch/nightly/cu102/libtorch-cxx11-abi-shared-with-deps-1.13.0.dev20220929%2Bcu102.zip && \
    unzip libtorch-gpu.zip && \
    rm libtorch-gpu.zip

# Clone Google Re2 (regex for thirdparty dependency)
RUN cd /opt && curl -L -o re2.zip https://github.com/google/re2/archive/refs/tags/2023-03-01.zip && \
    unzip re2.zip && \
    rm re2.zip && cd re2-2023-03-01 && mkdir build && cd build && \
    cmake .. && make -j${nproc} && make install


# Set environment variables for libtorch
ENV LIBTORCH_PATH=/opt/libtorch/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib:$LD_LIBRARY_PATH

# Clone matplotlib-cpp
RUN mkdir /opt/matplotlib-cpp && \
    cd /opt/matplotlib-cpp && \
    git clone https://github.com/lava/matplotlib-cpp.git

# Set the working directory in the container
WORKDIR /code

# Copy local files under the current dir into the container
COPY . /code

# Set the default command to bash
CMD ["/bin/bash"]

FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libopencv-dev curl make build-essential wget unzip mc nano

# RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/*

WORKDIR /src

COPY tensorflow/ /src/tensorflow/
RUN ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
RUN ./tensorflow/tensorflow/lite/tools/make/build_lib.sh

COPY deepseg.py *.h *.cc *.c Makefile CMakeLists.txt /src/
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt-get update \
    && apt-get install -y cmake

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y git
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get install --no-install-recommends -y libgles2-mesa-dev libgles2-mesa \
    && mkdir -p /etc/OpenCL/vendors \
    && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd

RUN mkdir -p cmake_build; cd cmake_build; cmake .. ; make -j4

FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
WORKDIR /src
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y libgles2-mesa libopencv-highgui-dev ocl-icd-opencl-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd
COPY --from=0 /src/cmake_build/backscrub /src/backscrub

ENTRYPOINT ["/src/backscrub"]
# CMD ["--bodypix-url", "http://bodypix:9000/"]

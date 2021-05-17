FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libopencv-dev curl make build-essential wget unzip mc nano

# RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/*

WORKDIR /src

COPY tensorflow/ /src/tensorflow/
RUN ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
RUN ./tensorflow/tensorflow/lite/tools/make/build_lib.sh

COPY deepseg.py *.h *.cc *.c Makefile /src/
RUN make

ENTRYPOINT ["/src/deepseg"]
# CMD ["--bodypix-url", "http://bodypix:9000/"]

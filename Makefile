
CFLAGS = -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -Wall -pthread
LDFLAGS = -lrt -ldl

# TensorFlow
TFBASE=../tensorflow
#TFBASE=/tensorflow_src
TFLITE=$(TFBASE)/tensorflow/lite/tools/make/
CFLAGS += -I $(TFBASE) -I $(TFLITE)/downloads/absl -I $(TFLITE)/downloads/flatbuffers/include
LDFLAGS += -L $(TFLITE)/gen/linux_x86_64/lib/ -ltensorflow-lite

# git clone -b v2.1.0  https://github.com/tensorflow/tensorflow $(TFBASE)
# cd $(TFBASE)/tensorflow/lite/tools/make
# ./download_dependencies.sh && ./build_lib.sh

# OpenCV
ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv)
    LDFLAGS += $(shell pkg-config --libs opencv)
else ifeq ($(shell pkg-config --exists opencv4; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv4)
    LDFLAGS += $(shell pkg-config --libs opencv4)
else
    $(error Couldn't find OpenCV)
endif

deepseg: deepseg.cc loopback.cc
	g++ $^ ${CFLAGS} ${LDFLAGS} -o $@

all: deepseg

clean:
	-rm deepseg

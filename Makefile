# this is licensed software, @see LICENSE file.
CFLAGS = -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -Wall -pthread
LDFLAGS = -lrt -ldl

# TensorFlow
TFBASE=tensorflow/
TFLITE=$(TFBASE)/tensorflow/lite/tools/make/
TFLIBS=$(TFLITE)/gen/linux_x86_64/lib/
CFLAGS += -I $(TFBASE) -I $(TFLITE)/downloads/absl -I $(TFLITE)/downloads/flatbuffers/include -ggdb
LDFLAGS += -L $(TFLIBS) -ltensorflow-lite -ldl

# OpenCV
ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv)
    LDFLAGS += $(shell pkg-config --libs opencv)
else ifeq ($(shell pkg-config --exists opencv4; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv4)
    LDFLAGS += $(shell pkg-config --libs opencv4)
else
    $(error Couldn\'t find OpenCV)
endif

deepseg: $(TFLIBS)/libtensorflow-lite.a deepseg.cc loopback.cc transpose_conv_bias.cc
	g++ $^ ${CFLAGS} ${LDFLAGS} -o $@

$(TFLIBS)/libtensorflow-lite.a: $(TFLITE)
	cd $(TFLITE) && ./download_dependencies.sh && ./build_lib.sh

$(TFLITE):
	git submodule update --init --recursive

all: deepseg

clean:
	-rm deepseg

tv: transparent_viewer.c
	g++ -o $@ $^ -lX11 -lGL $(CFLAGS) $(LDFLAGS)

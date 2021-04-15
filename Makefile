# this is licensed software, @see LICENSE file.

# OpenCV & Tensorflow recommended flags for performance..
CFLAGS = -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -Wall -pthread
LDFLAGS = -lrt -ldl

# Version
VERSION=$(shell git describe --all --long --always --dirty)
ifeq ($(VERSION),)
	VERSION=v0.2.0-no-git
endif

CFLAGS += -D DEEPSEG_VERSION=$(VERSION)

# TensorFlow
TFBASE=tensorflow
TFLITE=$(TFBASE)/tensorflow/lite/tools/make
TFLIBS=$(TFLITE)/gen/linux_x86_64/lib
TFCFLAGS += -I $(TFBASE) -I $(TFLITE)/downloads/absl -I $(TFLITE)/downloads/flatbuffers/include -ggdb
TFLDFLAGS += -L $(TFLIBS) -ltensorflow-lite -ldl

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

# Output folder
BIN=bin

# Default target
all: $(BIN) $(BIN)/deepseg

clean:
	-rm -rf $(BIN)

$(BIN):
	-mkdir -p $(BIN)

# Primary binary - special deps
$(BIN)/deepseg: $(TFLIBS)/libtensorflow-lite.a deepseg.cc loopback.cc transpose_conv_bias.cc
	g++ $^ ${CFLAGS} ${TFCFLAGS} ${LDFLAGS} ${TFLDFLAGS} -o $@

$(TFLIBS)/libtensorflow-lite.a: $(TFLITE)
	cd $(TFLITE) && ./download_dependencies.sh && ./build_lib.sh

$(TFLITE):
	git submodule update --init --recursive

# Single file test progs - OpenCV deps only
$(BIN)/%: %.cc
	g++ -o $@ $(CFLAGS) $^ $(LDFLAGS)

$(BIN)/%: %.cpp
	g++ -o $@ $(CFLAGS) $^ $(LDFLAGS)

$(BIN)/%: %.c
	g++ -o $@ $(CFLAGS) $^ $(LDFLAGS)

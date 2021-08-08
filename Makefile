# this is licensed software, @see LICENSE file.

# OpenCV & Tensorflow recommended flags for performance..
CFLAGS = -fPIC -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -Wall -pthread
LDFLAGS = -lrt -ldl

# Version
VERSION=$(shell git describe --all --long --always --dirty)
# default if outside a git repo..
ifeq ($(VERSION),)
	VERSION=v0.2.0-no-git
endif

CFLAGS += -D DEEPSEG_VERSION=$(VERSION) -I.

# TensorFlow
TENSORFLOW=tensorflow
TFLITE=$(TENSORFLOW)/tensorflow/lite/tools/make
TFDOWN=$(TFLITE)/downloads/cpuinfo
TFLIBS=$(TFLITE)/gen/linux_x86_64/lib
TFCFLAGS += -I $(TENSORFLOW) -I $(TFLITE)/downloads/absl -I $(TFLITE)/downloads/flatbuffers/include -ggdb
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

# Primary binaries - special deps
$(BIN)/deepseg: app/deepseg.cc $(BIN)/libbackscrub.a $(BIN)/libvideoio.a $(TFLIBS)/libtensorflow-lite.a
	g++ $^ ${CFLAGS} ${TFCFLAGS} ${LDFLAGS} ${TFLDFLAGS} -o $@

# Backscrub library, must be linked with libtensorflow-lite.a
$(BIN)/libbackscrub.a: $(BIN)/libbackscrub.o $(BIN)/transpose_conv_bias.o
	ar rv $@ $^

# Video I/O library - this is a Linux/v4l2loopback only target for now but replaceable later..
$(BIN)/libvideoio.a: $(BIN)/loopback.o
	ar rv $@ $^

# Compile rules for various source directories
$(BIN)/%.o: lib/%.cc $(TFDOWN)
	g++ $< ${CFLAGS} ${TFCFLAGS} -c -o $@

$(BIN)/%.o: videoio/%.cc $(TFDOWN)
	g++ $< ${CFLAGS} ${TFCFLAGS} -c -o $@

$(BIN)/%.o: app/%.cc $(TFDOWN)
	g++ $< ${CFLAGS} ${TFCFLAGS} -c -o $@

# As cloned, TFLite needs building into a static library, as per:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/make
# we split this into download (pre-compile) and build (pre-link)
$(TFDOWN): $(TFLITE)
	cd $(TFLITE) && ./download_dependencies.sh

$(TFLIBS)/libtensorflow-lite.a: $(TFDOWN)
	cd $(TFLITE) && ./build_lib.sh

$(TFLITE):
	git submodule update --init --recursive

# Single file test progs - OpenCV deps only
$(BIN)/%: %.cc $(BIN)
	g++ -o $@ $(CFLAGS) $< $(LDFLAGS)

$(BIN)/%: %.cpp $(BIN)
	g++ -o $@ $(CFLAGS) $< $(LDFLAGS)

$(BIN)/%: %.c $(BIN)
	g++ -o $@ $(CFLAGS) $< $(LDFLAGS)

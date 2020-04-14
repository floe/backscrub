TFBASE=../tensorflow.git/
TFLITE=$(TFBASE)/tensorflow/lite/tools/make/

# git clone -b v2.1.0  https://github.com/tensorflow/tensorflow $(TFBASE)
# cd $(TFBASE)/tensorflow/lite/tools/make
# ./download_dependencies.sh && ./build_lib.sh

deepseg: deepseg.cc loopback.cc
	g++ $^ -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -I $(TFLITE)/downloads/absl -I $(TFBASE) -I $(TFLITE)/downloads/flatbuffers/include/ -L $(TFLITE)/gen/linux_x86_64/lib/ $(shell pkg-config --libs --cflags opencv) -Wall -ltensorflow-lite -lrt -ldl -pthread -o $@

clean:
	-rm deepseg

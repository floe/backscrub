/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tested against tensorflow lite v2.1.0 (static library)

#include <unistd.h>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio/videoio_c.h>

#include "loopback.h"
#include "transpose_conv_bias.h"

// OpenCV helper functions
cv::Mat convert_rgb_to_yuyv( cv::Mat input ) {
	cv::Mat tmp;
	cv::cvtColor(input,tmp,CV_RGB2YUV);
	std::vector<cv::Mat> yuv;
	cv::split(tmp,yuv);
	cv::Mat yuyv(tmp.rows, tmp.cols, CV_8UC2);
	uint8_t* outdata = (uint8_t*)yuyv.data;
	uint8_t* ydata = (uint8_t*)yuv[0].data;
	uint8_t* udata = (uint8_t*)yuv[1].data;
	uint8_t* vdata = (uint8_t*)yuv[2].data;
	for (unsigned int i = 0; i < yuyv.total(); i += 2) {
		uint8_t u = (uint8_t)(((int)udata[i]+(int)udata[i+1])/2);
		uint8_t v = (uint8_t)(((int)vdata[i]+(int)vdata[i+1])/2);
		outdata[2*i+0] = ydata[i+0];
		outdata[2*i+1] = v;
		outdata[2*i+2] = ydata[i+1];
		outdata[2*i+3] = u;
	}
	return yuyv;
}

// Tensorflow Lite helper functions
using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
	fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

std::unique_ptr<Interpreter> interpreter;

cv::Mat getTensorMat(int tnum, int debug) {

	TfLiteType t_type = interpreter->tensor(tnum)->type;
	TFLITE_MINIMAL_CHECK(t_type == kTfLiteFloat32);

	TfLiteIntArray* dims = interpreter->tensor(tnum)->dims;
	if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
	TFLITE_MINIMAL_CHECK(dims->data[0] == 1);
	
	int h = dims->data[1];
	int w = dims->data[2];
	int c = dims->data[3];

	float* p_data = interpreter->typed_tensor<float>(tnum);
	TFLITE_MINIMAL_CHECK(p_data != nullptr);

	return cv::Mat(h,w,CV_32FC(c),p_data);
}

// deeplabv3 classes
std::vector<std::string> labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" };

// threaded capture shared state
typedef struct {
	cv::VideoCapture *cap;
	cv::Mat *grab;
	cv::Mat *raw;
	int64 cnt;
	pthread_mutex_t lock;
} capinfo_t;

// capture thread function
void *grab_thread(void *arg) {
	capinfo_t *ci = (capinfo_t *)arg;
	bool done = false;
	// while we have a grab frame.. grab frames
	while (!done) {
		ci->cap->grab();
		pthread_mutex_lock(&ci->lock);
		if (ci->grab!=NULL)
			ci->cap->retrieve(*ci->grab);
		else
			done = true;
		ci->cnt++;
		pthread_mutex_unlock(&ci->lock);
	}
	return NULL;
}

int main(int argc, char* argv[]) {

	printf("deepseg v0.1.0\n");
	printf("(c) 2020 by floe@butterbrot.org\n");
	printf("https://github.com/floe/deepseg\n");

	int debug  = 0;
	int threads= 2;
	int width  = 640;
	int height = 480;
	const char *back = "background.png";
	const char *vcam = "/dev/video0";
	const char *ccam = "/dev/video1";

	const char* modelname = "deeplabv3_257_mv_gpu.tflite";

	for (int arg=1; arg<argc; arg++) {
		if (strncmp(argv[arg], "-?", 2)==0) {
			fprintf(stderr, "usage: deepseg [-?] [-d] [-c <capture:/dev/video1>] [-v <vcam:/dev/video0>] [-w <width:640>] [-h <height:480>] [-t <threads:2>] [-b <background.png>]\n");
			exit(0);
		} else if (strncmp(argv[arg], "-d", 2)==0) {
			++debug;
		} else if (strncmp(argv[arg], "-v", 2)==0) {
			vcam = argv[++arg];
		} else if (strncmp(argv[arg], "-c", 2)==0) {
			ccam = argv[++arg];
		} else if (strncmp(argv[arg], "-b", 2)==0) {
			back = argv[++arg];
		} else if (strncmp(argv[arg], "-m", 2)==0) {
			modelname = argv[++arg];
		} else if (strncmp(argv[arg], "-w", 2)==0) {
			sscanf(argv[++arg], "%d", &width);
		} else if (strncmp(argv[arg], "-h", 2)==0) {
			sscanf(argv[++arg], "%d", &height);
		} else if (strncmp(argv[arg], "-t", 2)==0) {
			sscanf(argv[++arg], "%d", &threads);
		}
	}
	printf("debug:  %d\n", debug);
	printf("ccam:   %s\n", ccam);
	printf("vcam:   %s\n", vcam);
	printf("width:  %d\n", width);
	printf("height: %d\n", height);
	printf("back:   %s\n", back);
	printf("threads:%d\n", threads);

	cv::Mat bg = cv::imread(back);
	cv::resize(bg,bg,cv::Size(width,height));
	bg = convert_rgb_to_yuyv( bg );
	int lbfd = loopback_init(vcam,width,height,debug);

	cv::VideoCapture cap(ccam, CV_CAP_V4L2);
	TFLITE_MINIMAL_CHECK(cap.isOpened());

	cap.set(CV_CAP_PROP_FRAME_WIDTH,  width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cap.set(CV_CAP_PROP_FOURCC, *((uint32_t*)"YUYV"));
	cap.set(CV_CAP_PROP_CONVERT_RGB, false);

	// Load model
	std::unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile(modelname);
	TFLITE_MINIMAL_CHECK(model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	// custom op for Google Meet network
	resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
	InterpreterBuilder builder(*model, resolver);
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

	// set interpreter params
	interpreter->SetNumThreads(threads);
	interpreter->SetAllowFp16PrecisionForFp32(true);

	// get input and output tensor as cv::Mat
	cv::Mat  input = getTensorMat(interpreter->inputs ()[0],debug);
 	cv::Mat output = getTensorMat(interpreter->outputs()[0],debug);
	float ratio = (float)input.cols/(float) input.rows;

	// initialize mask and square ROI in center
	cv::Rect roidim = cv::Rect((width-height/ratio)/2,0,height/ratio,height);
	cv::Mat mask = cv::Mat::ones(height,width,CV_8UC1);
	cv::Mat mroi = mask(roidim);

	// erosion/dilation element
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(5,5) );

	const int cnum = labels.size();
	const int pers = std::find(labels.begin(),labels.end(),"person") - labels.begin();

	// kick off separate grabber thread to keep OpenCV/FFMpeg happy (or it lags badly)
	pthread_t grabber;
	cv::Mat buf1;
	cv::Mat buf2;
	capinfo_t capinfo = { &cap, &buf1, &buf2, 0, PTHREAD_MUTEX_INITIALIZER };
	if (pthread_create(&grabber, NULL, grab_thread, &capinfo)) {
		perror("creating grabber thread");
		exit(1);
	}
	// wait for first frame
	while (0==capinfo.cnt)
		usleep(1000);

	while (true) {

		int e1 = cv::getTickCount();

		// switch buffer pointers in capture thread
		pthread_mutex_lock(&capinfo.lock);
		cv::Mat *tmat = capinfo.grab;
		capinfo.grab = capinfo.raw;
		capinfo.raw = tmat;
		pthread_mutex_unlock(&capinfo.lock);
		// we can now guarantee capinfo.raw will remain unchanged while we process it..
		cv::Mat raw = (*capinfo.raw);
		if (raw.rows == 0 || raw.cols == 0) continue; // sanity check

		// map ROI
		cv::Mat roi = raw(roidim);

		// resize ROI to input size
		cv::Mat in_u8_yuv, in_u8_rgb;
		cv::resize(roi,in_u8_yuv,cv::Size(input.cols,input.rows));
		cv::cvtColor(in_u8_yuv,in_u8_rgb,CV_YUV2RGB_YUYV);
		// TODO: can convert directly to float?

		// convert to float and normalize values to [-1;1]
		in_u8_rgb.convertTo(input,CV_32FC3,1.0/128.0,-1.0);

		// Run inference
		TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

		// create Mat for small mask
		cv::Mat ofinal(output.rows,output.cols,CV_8UC1);
		float* tmp = (float*)output.data;
		uint8_t* out = (uint8_t*)ofinal.data;

		// find class with maximum probability
		if (strstr(modelname,"deeplab"))
		for (unsigned int n = 0; n < output.total(); n++) {
			float maxval = -10000; int maxpos = 0;
			for (int i = 0; i < cnum; i++) {
				if (tmp[n*cnum+i] > maxval) {
					maxval = tmp[n*cnum+i];
					maxpos = i;
				}
			}
			// set mask to 0 where class == person
			out[n] = (maxpos==pers ? 0 : 255);
		}

		// threshold probability
		if (strstr(modelname,"body-pix"))
		for (unsigned int n = 0; n < output.total(); n++) {
			// FIXME: hardcoded threshold
			if (tmp[n] > 0.65) out[n] = 0; else out[n] = 255;
		}

		// Google Meet segmentation network
		if (strstr(modelname,"segm_")) {
			// FIXME: not working yet
			/* 256 x 144 x 2 tensor for the full model or 160 x 96 x 2
			 * tensor for the light model with masks for background
			 * (channel 0) and person (channel 1) where values are in
			 * range [MIN_FLOAT, MAX_FLOAT] and user has to apply
			 * softmax across both channels to yield foreground
			 * probability in [0.0, 1.0]. */
		for (unsigned int n = 0; n < output.total(); n++) {
			float exp0 = expf(tmp[2*n  ]);
			float exp1 = expf(tmp[2*n+1]);
			float p0 = exp0 / (exp0+exp1);
			float p1 = exp1 / (exp0+exp1);
			if (p0 < p1) out[n] = 0; else out[n] = 255;
		}
		}

		// denoise
		cv::Mat tmpbuf;
		cv::dilate(ofinal,tmpbuf,element);
		cv::erode(tmpbuf,ofinal,element);

		// scale up into full-sized mask
		cv::resize(ofinal,mroi,cv::Size(raw.rows/ratio,raw.rows));

		// copy background over raw cam image using mask
		bg.copyTo(raw,mask);

		// write frame to v4l2loopback
		int framesize = raw.step[0]*raw.rows;
		while (framesize > 0) {
			int ret = write(lbfd,raw.data,framesize);
			TFLITE_MINIMAL_CHECK(ret > 0);
			framesize -= ret;
		}

		if (!debug) { printf("."); fflush(stdout); continue; }

		int e2 = cv::getTickCount();
		float t = (e2-e1)/cv::getTickFrequency();
		printf("FPS: %5.2f\r",1.0/t);
		fflush(stdout);
		if (debug < 2) continue;

		cv::Mat test;
		cv::cvtColor(raw,test,CV_YUV2BGR_YUYV);
		cv::imshow("output.png",test);
		if (cv::waitKey(1) == 'q') break;
	}
	pthread_mutex_lock(&capinfo.lock);
	capinfo.grab = NULL;
	pthread_mutex_unlock(&capinfo.lock);

  return 0;
}


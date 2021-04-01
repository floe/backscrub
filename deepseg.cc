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
#include <chrono>
#include <string>

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

int fourCcFromString(const std::string& in)
{
	if (in.empty())
		return 0;

	if (in.size() <= 4)
	{
		// fourcc codes are up to 4 bytes long, right-space-padded and upper-case
		// c.f. http://ffmpeg.org/doxygen/trunk/isom_8c-source.html and
		// c.f. https://www.fourcc.org/codecs.php
		std::array<uint8_t, 4> a = {' ', ' ', ' ', ' '};
		for (size_t i = 0; i < in.size(); ++i)
			a[i] = ::toupper(in[i]);
		return cv::VideoWriter::fourcc(a[0], a[1], a[2], a[3]);
	}
	else if (in.size() == 8)
	{
		// Most people seem to agree on 0x47504A4D being the fourcc code of "MJPG", not the literal translation
		// 0x4D4A5047. This is also what ffmpeg expects.
		return std::stoi(in, nullptr, 16);
	}
	return 0;
}

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
const std::vector<std::string> labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" };
// label number of "person" for DeepLab v3+ model
const size_t cnum = labels.size();
const size_t pers = std::distance(labels.begin(), std::find(labels.begin(),labels.end(),"person"));

// timing helpers
typedef std::chrono::high_resolution_clock::time_point timestamp_t;
typedef struct {
	timestamp_t bootns;
	timestamp_t lastns;
	timestamp_t waitns;
	timestamp_t lockns;
	timestamp_t copyns;
	timestamp_t openns;
	timestamp_t tfltns;
	timestamp_t maskns;
	timestamp_t postns;
	timestamp_t v4l2ns;
	// these are already converted to ns
	long grabns;
	long retrns;
} timinginfo_t;

timestamp_t timestamp() {
	return std::chrono::high_resolution_clock::now();
}
long diffnanosecs(timestamp_t t1, timestamp_t t2) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t2).count();
}

// threaded capture shared state
typedef struct {
	cv::VideoCapture *cap;
	cv::Mat *grab;
	cv::Mat *raw;
	int64 cnt;
	timinginfo_t *pti;
	pthread_mutex_t lock;
} capinfo_t;

typedef struct {
	const char *modelname;
	size_t threads;
	size_t width;
	size_t height;
	int debug;
	std::unique_ptr<tflite::FlatBufferModel> model;
	cv::Mat input;
	cv::Mat output;
	cv::Rect roidim;
	cv::Mat mask;
	cv::Mat mroi;
	cv::Mat raw;
	cv::Mat ofinal;
	cv::Mat element;
	float ratio;
} calcinfo_t;

// capture thread function
void *grab_thread(void *arg) {
	capinfo_t *ci = (capinfo_t *)arg;
	bool done = false;
	// while we have a grab frame.. grab frames
	while (!done) {
		timestamp_t ts = timestamp();
		ci->cap->grab();
		long ns = diffnanosecs(timestamp(),ts);
		pthread_mutex_lock(&ci->lock);
		ci->pti->grabns = ns;
		if (ci->grab!=NULL) {
			ts = timestamp();
			ci->cap->retrieve(*ci->grab);
			ci->pti->retrns = diffnanosecs(timestamp(),ts);
		} else {
			done = true;
		}
		ci->cnt++;
		pthread_mutex_unlock(&ci->lock);
	}
	return NULL;
}

void init_tensorflow(calcinfo_t &info) {
	// Load model
	info.model = tflite::FlatBufferModel::BuildFromFile(info.modelname);
	TFLITE_MINIMAL_CHECK(info.model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	// custom op for Google Meet network
	resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
	InterpreterBuilder builder(*info.model, resolver);
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

	// set interpreter params
	interpreter->SetNumThreads(info.threads);
	interpreter->SetAllowFp16PrecisionForFp32(true);

	// get input and output tensor as cv::Mat
	info.input = getTensorMat(interpreter->inputs ()[0],info.debug);
	info.output = getTensorMat(interpreter->outputs()[0],info.debug);
	info.ratio = (float)info.input.cols/(float) info.input.rows;

	// initialize mask and square ROI in center
	info.roidim = cv::Rect((info.width-info.height/info.ratio)/2,0,info.height/info.ratio,info.height);
	info.mask = cv::Mat::ones(info.height,info.width,CV_8UC1);
	info.mroi = info.mask(info.roidim);

	// erosion/dilation element
	info.element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(5,5) );

	// create Mat for small mask
	info.ofinal = cv::Mat(info.output.rows,info.output.cols,CV_8UC1);
}

void calc_mask(calcinfo_t &info, timinginfo_t &ti) {
	// map ROI
	cv::Mat roi = info.raw(info.roidim);

	// resize ROI to input size
	cv::Mat in_u8_bgr, in_u8_rgb;
	cv::resize(roi,in_u8_bgr,cv::Size(info.input.cols,info.input.rows));
	cv::cvtColor(in_u8_bgr,in_u8_rgb,CV_BGR2RGB);
	// TODO: can convert directly to float?

	// bilateral filter to reduce noise
	if (1) {
		cv::Mat filtered;
		cv::bilateralFilter(in_u8_rgb,filtered,5,100.0,100.0);
		in_u8_rgb = filtered;
	}

	// convert to float and normalize values to [-1;1]
	in_u8_rgb.convertTo(info.input,CV_32FC3,1.0/128.0,-1.0);
	ti.openns=timestamp();


	// Run inference
	TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
	ti.tfltns=timestamp();

	float* tmp = (float*)info.output.data;
	uint8_t* out = (uint8_t*)info.ofinal.data;

	// find class with maximum probability
	if (strstr(info.modelname,"deeplab")) {
		for (unsigned int n = 0; n < info.output.total(); n++) {
			float maxval = -10000; size_t maxpos = 0;
			for (size_t i = 0; i < cnum; i++) {
				if (tmp[n*cnum+i] > maxval) {
					maxval = tmp[n*cnum+i];
					maxpos = i;
				}
			}
			// set mask to 0 where class == person
			uint8_t val = (maxpos==pers ? 0 : 255);
			out[n] = (val & 0xE0) | (out[n] >> 3);
		}
	}

	// threshold probability
	if (strstr(info.modelname,"body-pix")) {
		for (unsigned int n = 0; n < info.output.total(); n++) {
			// FIXME: hardcoded threshold
			uint8_t val = (tmp[n] > 0.65 ? 0 : 255);
			out[n] = (val & 0xE0) | (out[n] >> 3);
		}
	}

	// Google Meet segmentation network
	if (strstr(info.modelname,"segm_")) {
		/* 256 x 144 x 2 tensor for the full model or 160 x 96 x 2
		 * tensor for the light model with masks for background
		 * (channel 0) and person (channel 1) where values are in
		 * range [MIN_FLOAT, MAX_FLOAT] and user has to apply
		 * softmax across both channels to yield foreground
		 * probability in [0.0, 1.0]. */
		for (unsigned int n = 0; n < info.output.total(); n++) {
			float exp0 = expf(tmp[2*n  ]);
			float exp1 = expf(tmp[2*n+1]);
			float p0 = exp0 / (exp0+exp1);
			float p1 = exp1 / (exp0+exp1);
			uint8_t val = (p0 < p1 ? 0 : 255);
			out[n] = (val & 0xE0) | (out[n] >> 3);
		}
	}
	ti.maskns=timestamp();

	// denoise
	cv::Mat tmpbuf;
	cv::dilate(info.ofinal,tmpbuf,info.element);
	cv::erode(tmpbuf,info.ofinal,info.element);

	// scale up into full-sized mask
	cv::resize(info.ofinal,info.mroi,cv::Size(info.raw.rows/info.ratio,info.raw.rows));
}

int main(int argc, char* argv[]) {

	printf("deepseg v0.2.0\n");
	printf("(c) 2021 by floe@butterbrot.org\n");
	printf("https://github.com/floe/deepbacksub\n");
	timinginfo_t ti;
	ti.bootns = timestamp();
	int debug  = 0;
	bool showProgress = false;
	size_t threads= 2;
	size_t width  = 640;
	size_t height = 480;
	const char *back = nullptr; // "images/background.png";
	const char *vcam = "/dev/video0";
	const char *ccam = "/dev/video1";
	bool flipHorizontal = false;
	bool flipVertical   = false;
	int fourcc = 0;

	const char* modelname = "models/segm_full_v679.tflite";

	bool showUsage = false;
	for (int arg=1; arg<argc; arg++) {
		bool hasArgument = arg+1 < argc;
		if (strncmp(argv[arg], "-?", 2)==0) {
			showUsage = true;
		} else if (strncmp(argv[arg], "-d", 2)==0) {
			++debug;
		} else if (strncmp(argv[arg], "-p", 2)==0) {
			showProgress = true;
		} else if (strncmp(argv[arg], "-H", 2)==0) {
			flipHorizontal = !flipHorizontal;
		} else if (strncmp(argv[arg], "-V", 2)==0) {
			flipVertical = !flipVertical;
		} else if (strncmp(argv[arg], "-v", 2)==0) {
			if (hasArgument) {
				vcam = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-c", 2)==0) {
			if (hasArgument) {
				ccam = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-b", 2)==0) {
			if (hasArgument) {
				back = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-m", 2)==0) {
			if (hasArgument) {
				modelname = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-w", 2)==0) {
			if (hasArgument && sscanf(argv[++arg], "%zu", &width)) {
				if (!width) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-h", 2)==0) {
			if (hasArgument && sscanf(argv[++arg], "%zu", &height)) {
				if (!height) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-f", 2)==0) {
			if (hasArgument) {
				fourcc = fourCcFromString(argv[++arg]);
				if (!fourcc) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-t", 2)==0) {
			if (hasArgument && sscanf(argv[++arg], "%zu", &threads)) {
				if (!threads) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		}
	}

	if (showUsage) {
		fprintf(stderr, "\n");
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "  deepseg [-?] [-d] [-p] [-c <capture>] [-v <virtual>] [-w <width>] [-h <height>]\n");
		fprintf(stderr, "    [-t <threads>] [-b <background>] [-m <modell>]\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "-?            Display this usage information\n");
		fprintf(stderr, "-d            Increase debug level\n");
		fprintf(stderr, "-p            Show progress bar\n");
		fprintf(stderr, "-c            Specify the video source (capture) device\n");
		fprintf(stderr, "-v            Specify the video target (sink) device\n");
		fprintf(stderr, "-w            Specify the video stream width\n");
		fprintf(stderr, "-h            Specify the video stream height\n");
		fprintf(stderr, "-f            Specify the camera video format, i.e. MJPG or 47504A4D.\n");
		fprintf(stderr, "-t            Specify the number of threads used for processing\n");
		fprintf(stderr, "-b            Specify the background image\n");
		fprintf(stderr, "-m            Specify the TFLite model used for segmentation\n");
		fprintf(stderr, "-H            Mirror the output horizontally\n");
		fprintf(stderr, "-V            Mirror the output vertically\n");
		exit(1);
	}

	printf("debug:  %d\n", debug);
	printf("ccam:   %s\n", ccam);
	printf("vcam:   %s\n", vcam);
	printf("width:  %zu\n", width);
	printf("height: %zu\n", height);
	printf("flip_h: %s\n", flipHorizontal ? "yes" : "no");
	printf("flip_v: %s\n", flipVertical ? "yes" : "no");
	printf("threads:%zu\n", threads);
	printf("back:   %s\n", back ? back : "(none)");
	printf("model:  %s\n\n", modelname);

	cv::Mat bg;
	if (back) {
		bg = cv::imread(back);
	}
	if (bg.empty()) {
		if (back) {
			printf("Warning: could not load background image, defaulting to green\n");
		}
		bg = cv::Mat(height,width,CV_8UC3,cv::Scalar(0,255,0));
	}
	cv::resize(bg,bg,cv::Size(width,height));

	int lbfd = loopback_init(vcam,width,height,debug);
	if(lbfd < 0) {
		fprintf(stderr, "Failed to initialize vcam device.\n");
		exit(1);
	}

	cv::VideoCapture cap(ccam, CV_CAP_V4L2);
	TFLITE_MINIMAL_CHECK(cap.isOpened());

	cap.set(CV_CAP_PROP_FRAME_WIDTH,  width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	if (fourcc)
		cap.set(CV_CAP_PROP_FOURCC, fourcc);
	cap.set(CV_CAP_PROP_CONVERT_RGB, true);

	calcinfo_t calcinfo = { modelname, threads, width, height, debug };
	init_tensorflow(calcinfo);

	// kick off separate grabber thread to keep OpenCV/FFMpeg happy (or it lags badly)
	pthread_t grabber;
	cv::Mat buf1;
	cv::Mat buf2;
	int64 oldcnt = 0;
	capinfo_t capinfo = { &cap, &buf1, &buf2, 0, &ti, PTHREAD_MUTEX_INITIALIZER };
	if (pthread_create(&grabber, NULL, grab_thread, &capinfo)) {
		perror("creating grabber thread");
		exit(1);
	}
	ti.lastns = timestamp();
	printf("Startup: %ldns\n", diffnanosecs(ti.lastns,ti.bootns));

	bool filterActive = true;

	// mainloop
	for(bool running = true; running; ) {
		// wait for next frame
		while (capinfo.cnt == oldcnt) usleep(10000);
		oldcnt = capinfo.cnt;
		int e1 = cv::getTickCount();
		ti.waitns=timestamp();

		// switch buffer pointers in capture thread
		pthread_mutex_lock(&capinfo.lock);
		ti.lockns=timestamp();
		cv::Mat *tmat = capinfo.grab;
		capinfo.grab = capinfo.raw;
		capinfo.raw = tmat;
		pthread_mutex_unlock(&capinfo.lock);
		// we can now guarantee capinfo.raw will remain unchanged while we process it..
		calcinfo.raw = *capinfo.raw;
		ti.copyns=timestamp();
		if (calcinfo.raw.rows == 0 || calcinfo.raw.cols == 0) continue; // sanity check

		if (filterActive) {
			// do background detection magic
			calc_mask(calcinfo, ti);

			// copy background over raw cam image using mask
			bg.copyTo(calcinfo.raw,calcinfo.mask);
		} // filterActive

		if (flipHorizontal && flipVertical) {
			cv::flip(calcinfo.raw,calcinfo.raw,-1);
		} else if (flipHorizontal) {
			cv::flip(calcinfo.raw,calcinfo.raw,1);
		} else if (flipVertical) {
			cv::flip(calcinfo.raw,calcinfo.raw,0);
		}
		ti.postns=timestamp();

		// write frame to v4l2loopback as YUYV
		calcinfo.raw = convert_rgb_to_yuyv(calcinfo.raw);
		int framesize = calcinfo.raw.step[0]*calcinfo.raw.rows;
		while (framesize > 0) {
			int ret = write(lbfd,calcinfo.raw.data,framesize);
			TFLITE_MINIMAL_CHECK(ret > 0);
			framesize -= ret;
		}
		ti.v4l2ns=timestamp();

		if (!debug) {
			if (showProgress) {
				printf(".");
				fflush(stdout);
			}
			continue;
		}

		// timing details..
		printf("wait:%9ld lock:%9ld [grab:%9ld retr:%9ld] copy:%9ld open:%9ld tflt:%9ld mask:%9ld post:%9ld v4l2:%9ld ",
			diffnanosecs(ti.waitns,ti.lastns),
			diffnanosecs(ti.lockns,ti.waitns),
			ti.grabns,
			ti.retrns,
			diffnanosecs(ti.copyns,ti.lockns),
			diffnanosecs(ti.openns,ti.copyns),
			diffnanosecs(ti.tfltns,ti.openns),
			diffnanosecs(ti.maskns,ti.tfltns),
			diffnanosecs(ti.postns,ti.maskns),
			diffnanosecs(ti.v4l2ns,ti.postns));

		int e2 = cv::getTickCount();
		float t = (e2-e1)/cv::getTickFrequency();
		printf("FPS: %5.2f\e[K\r",1.0/t);
		fflush(stdout);
		ti.lastns = timestamp();
		if (debug < 2) continue;

		cv::Mat test;
		cv::cvtColor(calcinfo.raw,test,CV_YUV2BGR_YUYV);
		cv::imshow("output.png",test);

		auto keyPress = cv::waitKey(1);
		switch(keyPress) {
			case 'q':
				running = false;
				break;
			case 's':
				filterActive = !filterActive;
				break;
			case 'h':
				flipHorizontal = !flipHorizontal;
				break;
			case 'v':
				flipVertical = !flipVertical;
				break;
		}
	}

	pthread_mutex_lock(&capinfo.lock);
	capinfo.grab = NULL;
	pthread_mutex_unlock(&capinfo.lock);

	printf("\n");
	return 0;
}

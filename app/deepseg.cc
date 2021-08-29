/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

#include <unistd.h>
#include <cstdio>
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "videoio/loopback.h"
#include "lib/libbackscrub.h"

// Due to weirdness in the C(++) preprocessor, we have to nest stringizing macros to ensure expansion
// http://gcc.gnu.org/onlinedocs/cpp/Stringizing.html, use _STR(<raw text or macro>).
#define __STR(X) #X
#define _STR(X) __STR(X)

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
	cv::cvtColor(input,tmp,cv::COLOR_RGB2YUV);
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

cv::Mat alpha_blend(cv::Mat srca, cv::Mat srcb, cv::Mat mask) {
	// alpha blend two (8UC3) source images using a mask (8UC1, 255=>srca, 0=>srcb), adapted from:
	// https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
	// "trust no-one" => we're about to mess with data pointers
	assert(srca.rows==srcb.rows);
	assert(srca.cols==srcb.cols);
	assert(mask.rows==srca.rows);
	assert(mask.cols==srca.cols);
	assert(srca.type()==CV_8UC3);
	assert(srcb.type()==CV_8UC3);
	assert(mask.type()==CV_8UC1);
	cv::Mat out = cv::Mat::zeros(srca.size(), srca.type());
	uint8_t *optr = (uint8_t*)out.data;
	uint8_t *aptr = (uint8_t*)srca.data;
	uint8_t *bptr = (uint8_t*)srcb.data;
	uint8_t *mptr = (uint8_t*)mask.data;
	int npix = srca.rows * srca.cols;
	for (int pix=0; pix<npix; ++pix) {
		// blending weights
		int aw=(int)(*mptr++), bw=255-aw;
		// blend each channel byte
		*optr++ = (uint8_t)(( (int)(*aptr++)*aw + (int)(*bptr++)*bw )/255);
		*optr++ = (uint8_t)(( (int)(*aptr++)*aw + (int)(*bptr++)*bw )/255);
		*optr++ = (uint8_t)(( (int)(*aptr++)*aw + (int)(*bptr++)*bw )/255);
	}
	return out;
}

// timing helpers
typedef std::chrono::high_resolution_clock::time_point timestamp_t;
typedef struct {
	timestamp_t bootns;
	timestamp_t lastns;
	timestamp_t lockns;
	timestamp_t copyns;
	timestamp_t prepns;
	timestamp_t maskns;
	timestamp_t postns;
	timestamp_t v4l2ns;
	timestamp_t grabns;
	timestamp_t retrns;
} timinginfo_t;

timestamp_t timestamp() {
	return std::chrono::high_resolution_clock::now();
}
long diffnanosecs(timestamp_t t1, timestamp_t t2) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t2).count();
}

// encapsulation of mask calculation logic and threading
class CalcMask final {
protected:
	enum class thread_state { RUNNING, DONE };
	volatile thread_state state;
	void *maskctx;
	timestamp_t t0;
	// buffers
	cv::Mat mask1;
	cv::Mat mask2;
	cv::Mat *mask_current;
	cv::Mat *mask_out;
	cv::Mat frame1;
	cv::Mat frame2;
	cv::Mat *frame_current;
	cv::Mat *frame_next;
	// thread synchronisation
	std::mutex lock_frame;
	std::mutex lock_mask;
	std::condition_variable condition_new_frame;
	bool new_frame;
	bool new_mask;
	std::thread thread;

	void run() {
		cv::Mat *raw_tmp;
		timestamp_t tloop;

		while(thread_state::RUNNING == this->state) {
			tloop = t0 = timestamp();
			/* actual handling */
			{
				std::unique_lock<std::mutex> hold(lock_frame);
				while (!new_frame) {
					condition_new_frame.wait(hold);
				}

				// change frame buffer pointer
				new_frame = false;
				raw_tmp = frame_next;
				frame_next = frame_current;
				frame_current = raw_tmp;
			}
			waitns=diffnanosecs(timestamp(), t0);
			t0 = timestamp();
			if(!bs_maskgen_process(maskctx, *frame_current, *mask_current)) {
				fprintf(stderr, "failed to process video frame\n");
				exit(1);
			}
			{
				std::unique_lock<std::mutex> hold(lock_mask);
				raw_tmp = mask_out;
				mask_out = mask_current;
				mask_current = raw_tmp;
				new_mask = true;
			}
			loopns = diffnanosecs(timestamp(), tloop);
		}
	}

	// timing callbacks
	static void onprep(void *ctx) {
		CalcMask *cls = (CalcMask *)ctx;
		cls->prepns=diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}
	static void oninfer(void *ctx) {
		CalcMask *cls = (CalcMask *)ctx;
		cls->tfltns=diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}
	static void onmask(void *ctx) {
		CalcMask *cls = (CalcMask *)ctx;
		cls->maskns=diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}

public:
	long waitns;
	long prepns;
	long tfltns;
	long maskns;
	long loopns;

	CalcMask(const char *modelname,
			 size_t threads,
			 size_t width,
			 size_t height) {
		maskctx = bs_maskgen_new(modelname,threads,width,height,nullptr,onprep,oninfer,onmask,this);
		if (!maskctx)
			throw "Could not create mask context";

		// Do all other initialization …
		frame_next = &frame1;
		frame_current = &frame2;
		mask_current = &mask1;
		mask_out = &mask2;
		new_frame = false;
		new_mask = false;
		state = thread_state::RUNNING;
		thread = std::thread(&CalcMask::run, this);
	}

	~CalcMask() {
		state = thread_state::DONE;
		thread.join();
		bs_maskgen_delete(maskctx);
	}

	void set_input_frame(cv::Mat &frame) {
		std::lock_guard<std::mutex> hold(lock_frame);
		*frame_next = frame.clone();
		new_frame = true;
		condition_new_frame.notify_all();
	}

	void get_output_mask(cv::Mat &out) {
		if (new_mask) {
			std::lock_guard<std::mutex> hold(lock_mask);
			out = mask_out->clone();
			new_mask = false;
		}
	}
};

static bool is_number(const std::string &s) {
	return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

int main(int argc, char* argv[]) try {

	printf("backscrub version %s\n", _STR(DEEPSEG_VERSION));
	printf("(c) 2021 by floe@butterbrot.org & contributors\n");
	printf("https://github.com/floe/backscrub\n");
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
	size_t blur_strength = 0;

	const char* modelname = "models/selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite";

	bool showUsage = false;
	for (int arg=1; arg<argc; arg++) {
		bool hasArgument = arg+1 < argc;
		if (strncmp(argv[arg], "-?", 2)==0) {
			showUsage = true;
		} else if (strncmp(argv[arg], "-d", 2)==0) {
			++debug;
		} else if (strncmp(argv[arg], "-s", 2)==0) {
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
		} else if (strncmp(argv[arg], "-p", 2)==0) {
			if (hasArgument) {
				std::string option = argv[++arg];
				std::string key = option.substr(0, option.find(":"));
				std::string value = option.substr(option.find(":")+1);
				if (key == "bgblur") {
					if (is_number(value)) {
					blur_strength = std::stoi(value);
					if (blur_strength % 2 == 0) {
						fprintf(stderr, "strength value must be odd\n");
						showUsage = true;
					}
					} else {
						printf("No strength value supplied, using default strength 25\n");
						blur_strength = 25;
					}
				} else {
					fprintf(stderr, "Unknown post-processing option: %s\n", option.c_str());
					showUsage = true;
				}
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
		} else {
			fprintf(stderr, "Unknown option: %s\n", argv[arg]);
		}
	}

	if (showUsage) {
		fprintf(stderr, "\n");
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "  deepseg [-?] [-d] [-p] [-c <capture>] [-v <virtual>] [-w <width>] [-h <height>]\n");
		fprintf(stderr, "    [-t <threads>] [-b <background>] [-m <modell>] [-p <option:value>]\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "-?            Display this usage information\n");
		fprintf(stderr, "-d            Increase debug level\n");
		fprintf(stderr, "-s            Show progress bar\n");
		fprintf(stderr, "-c            Specify the video source (capture) device\n");
		fprintf(stderr, "-v            Specify the video target (sink) device\n");
		fprintf(stderr, "-w            Specify the video stream width\n");
		fprintf(stderr, "-h            Specify the video stream height\n");
		fprintf(stderr, "-f            Specify the camera video format, i.e. MJPG or 47504A4D.\n");
		fprintf(stderr, "-t            Specify the number of threads used for processing\n");
		fprintf(stderr, "-b            Specify the background image\n");
		fprintf(stderr, "-m            Specify the TFLite model used for segmentation\n");
		fprintf(stderr, "-p            Add post-processing steps\n");
		fprintf(stderr, "-p bgblur:<strength>   Blur the video background\n");
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

	cv::VideoCapture cap(ccam, cv::CAP_V4L2);
	if(!cap.isOpened()) {
		perror("failed to open video device");
		exit(1);
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	if (fourcc)
		cap.set(cv::CAP_PROP_FOURCC, fourcc);
	cap.set(cv::CAP_PROP_CONVERT_RGB, true);

	cv::Mat mask(height, width, CV_8U);
	cv::Mat raw;
	CalcMask ai(modelname, threads, width, height);
	ti.lastns = timestamp();
	printf("Startup: %ldns\n", diffnanosecs(ti.lastns,ti.bootns));

	bool filterActive = true;

	// mainloop
	for(bool running = true; running; ) {
		// grab new frame from cam
		cap.grab();
		ti.grabns=timestamp();
		// copy new frame to buffer
		cap.retrieve(raw);
		ti.retrns=timestamp();
		ai.set_input_frame(raw);
		ti.copyns=timestamp();

		if (raw.rows == 0 || raw.cols == 0) continue; // sanity check

		if (blur_strength) {
			raw.copyTo(bg);
			cv::GaussianBlur(bg,bg,cv::Size(blur_strength,blur_strength),0);
		}
		ti.prepns = timestamp();

		if (filterActive) {
			// do background detection magic
			ai.get_output_mask(mask);

			// alpha blend background over foreground using mask
			raw = alpha_blend(bg, raw, mask);
		}
		ti.maskns = timestamp();

		if (flipHorizontal && flipVertical) {
			cv::flip(raw,raw,-1);
		} else if (flipHorizontal) {
			cv::flip(raw,raw,1);
		} else if (flipVertical) {
			cv::flip(raw,raw,0);
		}
		ti.postns=timestamp();

		// write frame to v4l2loopback as YUYV
		raw = convert_rgb_to_yuyv(raw);
		int framesize = raw.step[0]*raw.rows;
		while (framesize > 0) {
			int ret = write(lbfd,raw.data,framesize);
			if(ret <= 0) {
				perror("writing to loopback device");
				exit(1);
			}
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
		printf("main [grab:%9ld retr:%9ld copy:%9ld prep:%9ld mask:%9ld post:%9ld v4l2:%9ld FPS: %5.2f] ai: [wait:%9ld prep:%9ld tflt:%9ld mask:%9ld FPS: %5.2f] \e[K\r",
			diffnanosecs(ti.grabns,ti.lastns),
			diffnanosecs(ti.retrns,ti.grabns),
			diffnanosecs(ti.copyns,ti.retrns),
			diffnanosecs(ti.prepns,ti.copyns),
			diffnanosecs(ti.maskns,ti.prepns),
			diffnanosecs(ti.postns,ti.maskns),
			diffnanosecs(ti.v4l2ns,ti.postns),
			1e9/diffnanosecs(ti.v4l2ns,ti.lastns),
			ai.waitns,
			ai.prepns,
			ai.tfltns,
			ai.maskns,
			1e9/ai.loopns
		);
		fflush(stdout);
		ti.lastns = timestamp();
		if (debug < 2) continue;

		cv::Mat test;
		cv::cvtColor(raw,test,cv::COLOR_YUV2BGR_YUYV);
		cv::imshow("DeepSeg " _STR(DEEPSEG_VERSION),test);

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

	printf("\n");
	return 0;
} catch(const char* msg) {
	fprintf(stderr, "Error: %s\n", msg);
	return 1;
}

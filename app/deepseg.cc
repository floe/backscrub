/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "background.h"
#include "calcmask.h"
#include "utils.h"

#include "videoio/loopback.h"


// Ensure we have a default search location for resource files
#ifndef INSTALL_PREFIX
#error No INSTALL_PREFIX defined at compile time
#endif

#define DEBUG_WIN_NAME "Backscrub " _STR(DEEPSEG_VERSION) " ('?' for help)"

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

int main(int argc, char* argv[]) try {
	printf("%s version %s\n", argv[0], _STR(DEEPSEG_VERSION));
	printf("(c) 2021-2022 by floe@butterbrot.org & contributors\n");
	printf("https://github.com/floe/backscrub\n");

	timinginfo_t ti;
	ti.bootns = timestamp();

	int debug = 0;

	bool showUsage = false;

	bool showProgress = false;
	bool showBackground = true;
	bool showMask = true;
	bool showFPS = true;
	bool showHelp = false;

	size_t threads = 2;
	size_t width = 640;
	size_t height = 480;
	size_t blur_strength = 0;

	int fourcc = 0;

	bool flipHorizontal = false;
	bool flipVertical = false;
	bool multipass = false;

	std::string vcam = "/dev/video1";
	std::string ccam = "/dev/video0";
	std::optional<std::string> back;
	std::optional<std::string> modelname = "selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite";

	const std::vector<std::string> args(argv, argv + argc);

	for (int arg = 1; arg < argc; arg++) {
		bool hasArgument = arg + 1 < argc;

		if (args[arg] == "-?") {
			showUsage = true;
		} else if (args[arg] == "-d") {
			++debug;
		} else if (args[arg] == "-s") {
			showProgress = true;
		} else if (args[arg] == "-H") {
			flipHorizontal = !flipHorizontal;
		} else if (args[arg] == "-V") {
			flipVertical = !flipVertical;
		} else if (args[arg] == "-M") {
			multipass = !multipass;
		} else if (args[arg] == "-v") {
			if (hasArgument) {
				vcam = args[++arg];
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-c") {
			if (hasArgument) {
				ccam = args[++arg];
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-b") {
			if (hasArgument) {
				back = args[++arg];
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-m") {
			if (hasArgument) {
				modelname = args[++arg];
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-p") {
			if (hasArgument) {
				std::string option = args[++arg];
				std::string key = option.substr(0, option.find(":"));
				std::string value = option.substr(option.find(":") + 1);

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
		} else if (args[arg] == "-w") {
			if (hasArgument && sscanf(args[++arg].c_str(), "%zu", &width)) {
				if (!width) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-h") {
			if (hasArgument && sscanf(args[++arg].c_str(), "%zu", &height)) {
				if (!height) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-f") {
			if (hasArgument) {
				fourcc = fourCcFromString(args[++arg]);

				if (!fourcc) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (args[arg] == "-t") {
			if (hasArgument && sscanf(args[++arg].c_str(), "%zu", &threads)) {
				if (!threads) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else {
			fprintf(stderr, "Unknown option: %s\n", args[arg].c_str());
		}
	}

	if (showUsage) {
		fprintf(stderr, "\n");
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "  backscrub [-?] [-d] [-p] [-c <capture>] [-v <virtual>] [-w <width>] [-h <height>]\n");
		fprintf(stderr, "    [-t <threads>] [-b <background>] [-m <modell>] [-p <option:value>] [-H] [-V] [-M]\n");
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
		fprintf(stderr, "-b            Specify the background (any local or network OpenCV source) e.g.\n");
		fprintf(stderr, "                local:   backgrounds/total_landscaping.jpg\n");
		fprintf(stderr, "                network: https://git.io/JE9o5\n");
		fprintf(stderr, "-m            Specify the TFLite model used for segmentation\n");
		fprintf(stderr, "-p            Add post-processing steps\n");
		fprintf(stderr, "-p bgblur:<strength>   Blur the video background\n");
		fprintf(stderr, "-H            Mirror the output horizontally\n");
		fprintf(stderr, "-V            Mirror the output vertically\n");
		fprintf(stderr, "-M            Activate multi-pass filtering (for aspect ratio mismatch)\n");
		exit(1);
	}


	// permit unprefixed device names
	if (ccam.rfind("/dev/", 0) != 0)
		ccam = "/dev/" + ccam;

	if (vcam.rfind("/dev/", 0) != 0)
		vcam = "/dev/" + vcam;

	std::optional<std::string> s_model = resolve_path(modelname.value(), "models");
	std::optional<std::string> s_backg = back ? resolve_path(back.value(), "backgrounds") : std::nullopt;

	printf("debug:  %d\n", debug);
	printf("ccam:   %s\n", ccam.c_str());
	printf("vcam:   %s\n", vcam.c_str());
	printf("width:  %zu\n", width);
	printf("height: %zu\n", height);
	printf("flip_h: %s\n", flipHorizontal ? "yes" : "no");
	printf("flip_v: %s\n", flipVertical ? "yes" : "no");
	printf("multi:  %s\n", multipass ? "yes" : "no");
	printf("threads:%zu\n", threads);
	printf("back:   %s\n", s_backg ? s_backg.value().c_str() : "(none)");
	printf("model:  %s\n\n", s_model ? s_model.value().c_str() : "(none)");

	// No model - stop here
	if (!s_model) {
		printf("Error: unable to load specified model: %s\n", modelname.value().c_str());
		return 1;
	}

	// Create debug window early (ensures highgui is correctly initialised on this thread)
	if (debug > 1) {
		cv::namedWindow(DEBUG_WIN_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	}

	// Load background if specified
	auto pbk(s_backg ? load_background(s_backg.value(), debug) : nullptr);

	if (!pbk) {
		if (back) {
			printf("Warning: could not load background image, defaulting to green\n");
		}
	}

	// default green screen background
	cv::Mat bg = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 255, 0));

	int lbfd = loopback_init(vcam, width, height, debug);

	if(lbfd < 0) {
		fprintf(stderr, "Failed to initialize vcam device.\n");
		return 1;
	}

	on_scope_exit lbfd_closer([lbfd]() {
		loopback_free(lbfd);
	});

	cv::VideoCapture cap(ccam.c_str(), cv::CAP_V4L2);

	if(!cap.isOpened()) {
		perror("failed to open video capture device");
		return 1;
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	if (fourcc)
		cap.set(cv::CAP_PROP_FOURCC, fourcc);

	cap.set(cv::CAP_PROP_CONVERT_RGB, true);

	cv::Mat mask(height, width, CV_8U);
	cv::Mat raw;
	CalcMask ai(s_model.value(), threads, width, height);
	ti.lastns = timestamp();
	printf("Startup: %ldns\n", diffnanosecs(ti.lastns, ti.bootns));

	bool filterActive = true;

	// mainloop
	for(bool running = true; running; ) {
		// grab new frame from cam
		cap.grab();
		ti.grabns = timestamp();
		// copy new frame to buffer
		cap.retrieve(raw);
		ti.retrns = timestamp();
		ai.set_input_frame(raw, multipass);
		ti.copyns = timestamp();

		if (raw.rows == 0 || raw.cols == 0)
			continue; // sanity check

		if (filterActive) {
			// do background detection magic
			ai.get_output_mask(mask);

			// get background frame:
			// - specified source if set
			// - copy of input video if blur_strength != 0
			// - default green (initial value)
			bool canBlur = false;

			if (pbk) {
				if (grab_background(pbk, width, height, bg) < 0)
					throw "Failed to read background frame";

				canBlur = true;
			} else if (blur_strength) {
				raw.copyTo(bg);
				canBlur = true;
			}

			// blur frame if requested (unless it's just green)
			if (canBlur && blur_strength)
				cv::GaussianBlur(bg, bg, cv::Size(blur_strength, blur_strength), 0);

			ti.prepns = timestamp();
			// alpha blend background over foreground using mask
			raw = alpha_blend(bg, raw, mask);
		} else {
			ti.prepns = timestamp();
		}

		ti.maskns = timestamp();

		if (flipHorizontal && flipVertical) {
			cv::flip(raw, raw, -1);
		} else if (flipHorizontal) {
			cv::flip(raw, raw, 1);
		} else if (flipVertical) {
			cv::flip(raw, raw, 0);
		}

		ti.postns = timestamp();

		// write frame to v4l2loopback as YUYV
		raw = convert_rgb_to_yuyv(raw);
		int framesize = raw.step[0] * raw.rows;

		while (framesize > 0) {
			int ret = write(lbfd, raw.data, framesize);

			if(ret <= 0) {
				perror("writing to loopback device");
				exit(1);
			}

			framesize -= ret;
		}

		ti.v4l2ns = timestamp();

		if (!debug) {
			if (showProgress) {
				printf(".");
				fflush(stdout);
			}

			continue;
		}

		// timing details..
		double mfps = 1e9 / diffnanosecs(ti.v4l2ns, ti.lastns);
		double afps = 1e9 / ai.loopns;
		printf("main [grab:%9ld retr:%9ld copy:%9ld prep:%9ld mask:%9ld post:%9ld v4l2:%9ld FPS: %5.2f] ai: [wait:%9ld prep:%9ld tflt:%9ld mask:%9ld FPS: %5.2f] \e[K\r",
			diffnanosecs(ti.grabns, ti.lastns),
			diffnanosecs(ti.retrns, ti.grabns),
			diffnanosecs(ti.copyns, ti.retrns),
			diffnanosecs(ti.prepns, ti.copyns),
			diffnanosecs(ti.maskns, ti.prepns),
			diffnanosecs(ti.postns, ti.maskns),
			diffnanosecs(ti.v4l2ns, ti.postns),
			mfps,
			ai.waitns,
			ai.prepns,
			ai.tfltns,
			ai.maskns,
			afps
		);
		fflush(stdout);
		ti.lastns = timestamp();

		if (debug < 2)
			continue;

		cv::Mat test;
		cv::cvtColor(raw, test, cv::COLOR_YUV2BGR_YUYV);

		// frame rates at the bottom
		if (showFPS) {
			char status[80];
			snprintf(status, sizeof(status), "MainFPS: %5.2f AiFPS: %5.2f", mfps, afps);
			cv::putText(test, status, cv::Point(5, test.rows - 5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
		}

		// keyboard help
		if (showHelp) {
			static const std::string help[] = {
				"Keyboard help:",
				" q: quit",
				" s: switch filter on/off",
				" h: toggle horizontal flip",
				" v: toggle vertical flip",
				" f: toggle FPS display on/off",
				" b: toggle background display on/off",
				" m: toggle mask display on/off",
				" M: toggle multi-pass processing on/off",
				" ?: toggle this help text on/off"
			};

			for (int i = 0; i < sizeof(help) / sizeof(std::string); i++) {
				cv::putText(test, help[i], cv::Point(10, test.rows / 2 + i * 15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
			}
		}

		// background as pic-in-pic
		if (showBackground && pbk) {
			cv::Mat thumb;
			grab_thumbnail(pbk, thumb);

			if (!thumb.empty()) {
				cv::Rect r = cv::Rect(0, 0, thumb.cols, thumb.rows);
				cv::Mat tri = test(r);
				thumb.copyTo(tri);
				cv::rectangle(test, r, cv::Scalar(255, 255, 255));
			}
		}

		// mask as pic-in-pic
		if (showMask) {
			if (!mask.empty()) {
				cv::Mat smask, cmask;
				int mheight = mask.rows * 160 / mask.cols;
				cv::resize(mask, smask, cv::Size(160, mheight));
				cv::cvtColor(smask, cmask, cv::COLOR_GRAY2BGR);
				cv::Rect r = cv::Rect(width - 160, 0, 160, mheight);
				cv::Mat mri = test(r);
				cmask.copyTo(mri);
				cv::rectangle(test, r, cv::Scalar(255, 255, 255));
				cv::putText(test, "Mask", cv::Point(width - 155, 115), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
			}
		}

		cv::imshow(DEBUG_WIN_NAME, test);

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

			case 'f':
				showFPS = !showFPS;
				break;

			case 'b':
				showBackground = !showBackground;
				break;

			case 'm':
				showMask = !showMask;
				break;

			case 'M':
				multipass = !multipass;
				break;

			case '?':
				showHelp = !showHelp;
				break;
		}
	}

	printf("\n");
	return 0;
} catch(const char* msg) {
	fprintf(stderr, "Error: %s\n", msg);
	return 1;
}

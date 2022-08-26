/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

#include <unistd.h>
#include <cstdio>
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include <fstream>
#include <istream>
#include <regex>
#include <optional>
#include <utility>
#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "videoio/loopback.h"
#include "lib/libbackscrub.h"
#include "background.h"

// Temporary declaration of utility class until we merge experimental!
class on_scope_exit final {
private:
	const std::function<void()> dtor;
public:
	explicit inline on_scope_exit(const std::function<void()>& f) : dtor(f) {}
	on_scope_exit() = delete;
	on_scope_exit(const on_scope_exit&) = delete;
	inline ~on_scope_exit() {
		if(dtor) {
			dtor();
		}
	}
};

// Due to weirdness in the C(++) preprocessor, we have to nest stringizing macros to ensure expansion
// http://gcc.gnu.org/onlinedocs/cpp/Stringizing.html, use _STR(<raw text or macro>).
#define __STR(X) #X
#define _STR(X) __STR(X)

// Ensure we have a default search location for resource files
#ifndef INSTALL_PREFIX
#error No INSTALL_PREFIX defined at compile time
#endif

#define DEBUG_WIN_NAME "Backscrub " _STR(DEEPSEG_VERSION) " ('?' for help)"

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

// Parse a geometry specification
std::optional<std::pair<size_t, size_t>> geometryFromString(const std::string& in) {
	size_t w, h;
	if (sscanf(in.c_str(), "%zux%zu", &w, &h)!=2)
		return {};
	return std::pair<size_t, size_t>(w, h);
}

// OpenCV helper functions
cv::Mat convert_rgb_to_yuyv( cv::Mat input ) {
	cv::Mat tmp;
	cv::cvtColor(input, tmp, cv::COLOR_RGB2YUV);
	std::vector<cv::Mat> yuv;
	cv::split(tmp, yuv);
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
	assert(srca.rows == srcb.rows);
	assert(srca.cols == srcb.cols);
	assert(mask.rows == srca.rows);
	assert(mask.cols == srca.cols);
	assert(srca.type() == CV_8UC3);
	assert(srcb.type() == CV_8UC3);
	assert(mask.type() == CV_8UC1);
	cv::Mat out = cv::Mat::zeros(srca.size(), srca.type());
	uint8_t *optr = (uint8_t*)out.data;
	uint8_t *aptr = (uint8_t*)srca.data;
	uint8_t *bptr = (uint8_t*)srcb.data;
	uint8_t *mptr = (uint8_t*)mask.data;
	int npix = srca.rows * srca.cols;
	for (int pix = 0; pix < npix; ++pix) {
		// blending weights
		int aw = (int)(*mptr++), bw = 255-aw;
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
			waitns = diffnanosecs(timestamp(), t0);
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
		cls->prepns = diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}
	static void oninfer(void *ctx) {
		CalcMask *cls = (CalcMask *)ctx;
		cls->tfltns = diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}
	static void onmask(void *ctx) {
		CalcMask *cls = (CalcMask *)ctx;
		cls->maskns = diffnanosecs(timestamp(), cls->t0);
		cls->t0 = timestamp();
	}

public:
	long waitns;
	long prepns;
	long tfltns;
	long maskns;
	long loopns;

	CalcMask(const std::string& modelname,
			 size_t threads,
			 size_t width,
			 size_t height) {
		maskctx = bs_maskgen_new(modelname.c_str(), threads, width, height, nullptr, onprep, oninfer, onmask, this);
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
		// mark as done
		state = thread_state::DONE;
		// wake up processing thread
		new_frame = true;
		condition_new_frame.notify_all();
		// collect termination
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

std::optional<std::string> resolve_path(const std::string& provided, const std::string& type) {
	std::string result;
	// Check for network (URI) schema and return as-is
	// https://www.rfc-editor.org/rfc/rfc3986#section-3.1
	// however we require at least two chars in the scheme to allow driver letters to work on Windows..
	if (std::regex_match(provided, std::regex("^[[:alpha:]][[:alnum:]+-.]{1,}:.*$")))
		return provided;
	// We use std::ifstream to check we can open each test path read-only, in order:
	// 1. exactly what was provided
	if (std::ifstream(provided).good())
		return provided;
	// to emulate PATH search behaviour (rule of least surprise), we stop here if provided has path separators
	if (provided.find('/') != provided.npos)
		return {};
	// 2. BACKSCRUB_PATH prefixes if set
	if (getenv("BACKSCRUB_PATH") != nullptr) {
		// getline trick: https://stackoverflow.com/questions/5167625/splitting-a-c-stdstring-using-tokens-e-g
		std::istringstream bsp(getenv("BACKSCRUB_PATH"));
		while (std::getline(bsp, result, ':')) {
			result += "/" + type + "/" + provided;
			if (std::ifstream(result).good())
				return result;
		}
	}
	// 3. XDG standard data location
	result = getenv("XDG_DATA_HOME") ? getenv("XDG_DATA_HOME") : std::string() + getenv("HOME") + "/.local/share";
	result += "/backscrub/" + type + "/" + provided;
	if (std::ifstream(result).good())
		return result;
	// 4. prefixed with compile-time install path
	result = std::string() + _STR(INSTALL_PREFIX) + "/share/backscrub/" + type + "/" + provided;
	if (std::ifstream(result).good())
		return result;
	// 5. relative to current binary location
	// (https://stackoverflow.com/questions/933850/how-do-i-find-the-location-of-the-executable-in-c)
	char binloc[1024];
	ssize_t n = readlink("/proc/self/exe", binloc, sizeof(binloc));
	if (n > 0) {
		binloc[n] = 0;
		result = binloc;
		size_t pos = result.rfind('/');
		pos = result.rfind('/', pos-1);
		if (pos != result.npos) {
			result.erase(pos);
			result += "/share/backscrub/" + type + "/" + provided;
			if (std::ifstream(result).good())
				return result;
			// development folder?
			result.erase(pos);
			result += "/" + type + "/" + provided;
			if (std::ifstream(result).good())
				return result;
		}
	}
	return {};
}

int main(int argc, char* argv[]) try {

	printf("%s version %s (Tensorflow: build %s, run-time %s)\n", argv[0], _STR(DEEPSEG_VERSION), _STR(TF_VERSION), bs_tensorflow_version());
	printf("(c) 2021 by floe@butterbrot.org & contributors\n");
	printf("https://github.com/floe/backscrub\n");
	timinginfo_t ti;
	ti.bootns = timestamp();
	int debug = 0;
	bool showProgress = false;
	bool showBackground = true;
	bool showMask = true;
	bool showFPS = true;
	bool showHelp = false;
	size_t threads = 2;
	size_t width = 640;
	size_t height = 480;
	bool setWorH = false;
	std::optional<std::pair<size_t, size_t>> capGeo = {};
	std::optional<std::pair<size_t, size_t>> vidGeo = {};
	const char *back = nullptr;
	const char *vcam = "/dev/video1";
	const char *ccam = "/dev/video0";
	bool flipHorizontal = false;
	bool flipVertical = false;
	int fourcc = 0;
	size_t blur_strength = 0;

	const char* modelname = "selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite";

	bool showUsage = false;
	for (int arg = 1; arg < argc; arg++) {
		bool hasArgument = arg+1 < argc;
		if (strncmp(argv[arg], "-?", 2) == 0) {
			showUsage = true;
		} else if (strncmp(argv[arg], "-d", 2) == 0) {
			++debug;
		} else if (strncmp(argv[arg], "-s", 2) == 0) {
			showProgress = true;
		} else if (strncmp(argv[arg], "-H", 2) == 0) {
			flipHorizontal = !flipHorizontal;
		} else if (strncmp(argv[arg], "-V", 2) == 0) {
			flipVertical = !flipVertical;
		} else if (strncmp(argv[arg], "-v", 2) == 0) {
			if (hasArgument) {
				vcam = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-c", 2) == 0) {
			if (hasArgument) {
				ccam = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-b", 2) == 0) {
			if (hasArgument) {
				back = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-m", 2) == 0) {
			if (hasArgument) {
				modelname = argv[++arg];
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-p", 2) == 0) {
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
		// deprecated width/height switches (implicitly capture and virtual camera size)
		} else if (strncmp(argv[arg], "-w", 2) == 0) {
			if (hasArgument && sscanf(argv[++arg], "%zu", &width)) {
				if (!width) {
					showUsage = true;
				}
				setWorH = true;
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-h", 2) == 0) {
			if (hasArgument && sscanf(argv[++arg], "%zu", &height)) {
				if (!height) {
					showUsage = true;
				}
				setWorH = true;
			} else {
				showUsage = true;
			}
		// replacement geometry switches (separate capture and virtual camera size)
		} else if (strncmp(argv[arg], "--cg", 4) == 0) {
			if (hasArgument) {
				capGeo = geometryFromString(argv[++arg]);
				if (!capGeo)
					showUsage = true;
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "--vg", 4) == 0) {
			if (hasArgument) {
				vidGeo = geometryFromString(argv[++arg]);
				if (!vidGeo)
					showUsage = true;
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-f", 2) == 0) {
			if (hasArgument) {
				fourcc = fourCcFromString(argv[++arg]);
				if (!fourcc) {
					showUsage = true;
				}
			} else {
				showUsage = true;
			}
		} else if (strncmp(argv[arg], "-t", 2) == 0) {
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

	// prevent use of both deprecated and current switches
	if (setWorH && (capGeo || vidGeo)) {
		showUsage = true;
		fprintf(stderr, "Error: (DEPRECATED) -w/-h used in conjunction with --cg/--vg.\n");
	}
	// set capture device geometry from deprecated switches if not set already
	if (!capGeo) {
		capGeo = std::pair<size_t, size_t>(width, height);
	}
	if (showUsage) {
		fprintf(stderr, "\n");
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "  backscrub [-?] [-d] [-p] [-c <capture>] [-v <virtual>] [--cg <width>x<height>]\n");
		fprintf(stderr, "    [--vg <width>x<height>] [-t <threads>] [-b <background>] [-m <modell>] [-p <option:value>]\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "-?            Display this usage information\n");
		fprintf(stderr, "-d            Increase debug level\n");
		fprintf(stderr, "-s            Show progress bar\n");
		fprintf(stderr, "-c            Specify the video capture (source) device\n");
		fprintf(stderr, "-v            Specify the virtual camera (sink) device\n");
		fprintf(stderr, "-w            DEPRECATED: Specify the video stream width\n");
		fprintf(stderr, "-h            DEPRECATED: Specify the video stream height\n");
		fprintf(stderr, "--cg          Specify the capture device geometry as <width>x<height>\n");
		fprintf(stderr, "--vg          Specify the virtual camera geometry as <width>x<height>\n");
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
		exit(1);
	}

	std::string s_ccam(ccam);
	std::string s_vcam(vcam);
	// permit unprefixed device names
	if (s_ccam.rfind("/dev/", 0) != 0)
		s_ccam = "/dev/" + s_ccam;
	if (s_vcam.rfind("/dev/", 0) != 0)
		s_vcam = "/dev/" + s_vcam;
	std::optional<std::string> s_model = resolve_path(modelname, "models");
	std::optional<std::string> s_backg = back ? resolve_path(back, "backgrounds") : std::nullopt;
	// open capture early to resolve true geometry
	cv::VideoCapture cap(s_ccam.c_str(), cv::CAP_V4L2);
	if(!cap.isOpened()) {
		perror("failed to open capture device");
		exit(1);
	}
	// set fourcc (if specified) /before/ attempting to set geometry (@see issue146)
	if (fourcc)
		cap.set(cv::CAP_PROP_FOURCC, fourcc);
	cap.set(cv::CAP_PROP_FRAME_WIDTH,  capGeo.value().first);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, capGeo.value().second);
	cap.set(cv::CAP_PROP_CONVERT_RGB, true);
	std::optional<std::pair<size_t, size_t>> tmpGeo = std::pair<size_t, size_t>(
		(size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH),
		(size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
	);
	if (tmpGeo != capGeo) {
		fprintf(stderr, "Warning: capture device geometry changed from requested values.\n");
		capGeo = tmpGeo;
	}
	if (!vidGeo) {
		vidGeo = capGeo;
	}
	// aspect ratio changed? warn
	// NB: we calculate this way round to avoid comparing doubles..
	size_t expWidth = (size_t)((double)vidGeo.value().second * (double)capGeo.value().first/(double)capGeo.value().second);
	if (expWidth != vidGeo.value().first) {
		fprintf(stderr, "Warning: virtual camera aspect ratio does not match capture device.\n");
	}

	// dump settings..
	printf("debug:  %d\n", debug);
	printf("ccam:   %s\n", s_ccam.c_str());
	printf("vcam:   %s\n", s_vcam.c_str());
	printf("capGeo: %zux%zu\n", capGeo.value().first, capGeo.value().second);
	printf("vidGeo: %zux%zu\n", vidGeo.value().first, vidGeo.value().second);
	printf("flip_h: %s\n", flipHorizontal ? "yes" : "no");
	printf("flip_v: %s\n", flipVertical ? "yes" : "no");
	printf("threads:%zu\n", threads);
	printf("back:   %s => %s\n", back ? back : "(none)", s_backg ? s_backg.value().c_str() : "(none)");
	printf("model:  %s => %s\n\n", modelname ? modelname : "(none)", s_model ? s_model.value().c_str() : "(none)");

	// No model - stop here
	if (!s_model) {
		printf("Error: unable to load specified model: %s\n", modelname);
		exit(1);
	}

	// Create debug window early (ensures highgui is correctly initialised on this thread)
	if (debug > 1) {
		cv::namedWindow(DEBUG_WIN_NAME, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_EXPANDED);
	}

	// Load background if specified
	auto pbk(s_backg ? load_background(s_backg.value(), debug) : nullptr);
	if (!pbk) {
		if (s_backg) {
			printf("Warning: could not load background image, defaulting to green\n");
		}
	}
	// default green screen background (at capture true geometry)
	cv::Mat bg = cv::Mat(capGeo.value().second, capGeo.value().first, CV_8UC3, cv::Scalar(0, 255, 0));

	// Virtual camera (at specified geometry)
	int lbfd = loopback_init(s_vcam, vidGeo.value().first, vidGeo.value().second, debug);
	if(lbfd < 0) {
		fprintf(stderr, "Failed to initialize vcam device.\n");
		exit(1);
	}

	on_scope_exit lbfd_closer([lbfd]() {
		loopback_free(lbfd);
	});


	// Processing components, all at capture true geometry
	cv::Mat mask(capGeo.value().second, capGeo.value().first, CV_8U);
	cv::Mat raw;
	CalcMask ai(s_model.value(), threads, capGeo.value().first, capGeo.value().second);
	ti.lastns = timestamp();
	printf("Startup: %ldns\n", diffnanosecs(ti.lastns,ti.bootns));

	bool filterActive = true;

	// mainloop
	for(bool running = true; running; ) {
		// grab new frame from cam
		cap.grab();
		ti.grabns = timestamp();
		// copy new frame to buffer
		cap.retrieve(raw);
		ti.retrns = timestamp();
		ai.set_input_frame(raw);
		ti.copyns = timestamp();

		if (raw.rows == 0 || raw.cols == 0) continue; // sanity check

		if (filterActive) {
			// do background detection magic
			ai.get_output_mask(mask);

			// get background frame:
			// - specified source if set
			// - copy of input video if blur_strength != 0
			// - default green (initial value)
			bool canBlur = false;
			if (pbk) {
				if (grab_background(pbk, capGeo.value().first, capGeo.value().second, bg)<0)
					throw "Failed to read background frame";
				canBlur = true;
			} else if (blur_strength) {
				raw.copyTo(bg);
				canBlur = true;
			}
			// blur frame if requested (unless it's just green)
			if (canBlur && blur_strength)
				cv::GaussianBlur(bg,bg,cv::Size(blur_strength,blur_strength), 0);
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

		// scale to virtual camera geometry (if required)
		if (vidGeo != capGeo) {
			cv::resize(raw, raw, cv::Size(vidGeo.value().first,vidGeo.value().second));
		}
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
		double mfps = 1e9/diffnanosecs(ti.v4l2ns,ti.lastns);
		double afps = 1e9/ai.loopns;
		printf("main [grab:%9ld retr:%9ld copy:%9ld prep:%9ld mask:%9ld post:%9ld v4l2:%9ld FPS: %5.2f] ai: [wait:%9ld prep:%9ld tflt:%9ld mask:%9ld FPS: %5.2f] \e[K\r",
			diffnanosecs(ti.grabns,ti.lastns),
			diffnanosecs(ti.retrns,ti.grabns),
			diffnanosecs(ti.copyns,ti.retrns),
			diffnanosecs(ti.prepns,ti.copyns),
			diffnanosecs(ti.maskns,ti.prepns),
			diffnanosecs(ti.postns,ti.maskns),
			diffnanosecs(ti.v4l2ns,ti.postns),
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
		cv::cvtColor(raw,test,cv::COLOR_YUV2BGR_YUYV);
		// frame rates & sizes at the bottom
		if (showFPS) {
			char status[80];
			snprintf(status, sizeof(status), "MainFPS: %5.2f AiFPS: %5.2f (%zux%zu->%zux%zu)",
				mfps, afps, capGeo.value().first, capGeo.value().second, vidGeo.value().first, vidGeo.value().second);
			cv::putText(test, status, cv::Point(5, test.rows-5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
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
				" ?: toggle this help text on/off"
			};
			for (int i=0; i<sizeof(help)/sizeof(std::string); i++) {
				cv::putText(test, help[i], cv::Point(10,test.rows/2+i*15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255));
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
				cv::rectangle(test, r, cv::Scalar(255,255,255));
			}
		}
		// mask as pic-in-pic
		if (showMask) {
			if (!mask.empty()) {
				cv::Mat smask, cmask;
				int mheight = mask.rows*160/mask.cols;
				cv::resize(mask, smask, cv::Size(160, mheight));
				cv::cvtColor(smask, cmask, cv::COLOR_GRAY2BGR);
				cv::Rect r = cv::Rect(vidGeo.value().first-160, 0, 160, mheight);
				cv::Mat mri = test(r);
				cmask.copyTo(mri);
				cv::rectangle(test, r, cv::Scalar(255,255,255));
				cv::putText(test, "Mask", cv::Point(vidGeo.value().first-155,115), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255));
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

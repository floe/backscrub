/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

#include "libbackscrub.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <vector>

#include <math.h>

#include "transpose_conv_bias.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"


// Internal context structures
enum class modeltype_t {
	Unknown,
	BodyPix,
	DeepLab,
	GoogleMeetSegmentation,
	MLKitSelfie,
};

struct normalization_t {
	float scaling;
	float offset;
};

struct backscrub_rect_t {
	cv::Rect src;
	cv::Rect dst;

	backscrub_rect_t() = delete;
	backscrub_rect_t(const cv::Rect& _src, const cv::Rect& _dst) : src(_src), dst(_dst) {};
	backscrub_rect_t(const backscrub_rect_t& other) = default;
};

struct backscrub_point_t {
	size_t x;
	size_t y;

	backscrub_point_t() = delete;
	backscrub_point_t(size_t _x, size_t _y) : x(_x), y(_y) {};
	backscrub_point_t(const backscrub_point_t& other) = default;
};

struct backscrub_ctx_t {
	// Loaded inference model
	std::unique_ptr<tflite::FlatBufferModel> model;

	// Model interpreter instance
	std::unique_ptr<tflite::Interpreter> interpreter;

	// Specific model type & input normalization
	modeltype_t modeltype;
	normalization_t norm;

	// Optional callbacks with caller-provided context
	void (*ondebug)(void *ctx, const char *msg);
	void (*onprep)(void *ctx);
	void (*oninfer)(void *ctx);
	void (*onmask)(void *ctx);
	void *caller_ctx;

	cv::Rect img_dim;		// Image dimensions

	// Single step variables
	cv::Mat input;			// NN input tensors
	cv::Mat output;			// NN output tensors
	cv::Mat ofinal;			// NN output (post-processed mask)

	float src_ratio;		// Source image aspect ratio
	cv::Rect src_roidim;	// Source image rect of interest
	cv::Mat mask_region;	// Region of the final mask to operate on

	float net_ratio;		// NN input image aspect ratio
	cv::Rect net_roidim;	// NN input image rect of interest

	// Result stitching variables
	cv::Mat in_u8_bgr;

	cv::Size blur;			// Size of blur on final mask
	cv::Mat mask;			// Fully processed mask (full image)

	// Information about the regions to process
	std::vector<backscrub_rect_t> region_rects;
};

// Debug helper
#ifdef WIN32
// https://stackoverflow.com/questions/40159892/using-asprintf-on-windows
static int vasprintf(char **msgp, const char *fmt, va_list ap) {
	int len = _vscprintf(fmt, ap);

	if (len <= 0)
		return len;

	*msgp = (char *)malloc(len + 1);
	len = vsprintf_s(*msgp, len + 1, fmt, ap);

	if (len <= 0) {
		free(*msgp);
		return len;
	}

	return len;
}
#endif
static void _dbg(backscrub_ctx_t &ctx, const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	char *msg;

	if (ctx.ondebug && vasprintf(&msg, fmt, ap) > 0) {
		ctx.ondebug(ctx.caller_ctx, msg);
		free(msg);
	} else {
		vfprintf(stderr, fmt, ap);
	}

	va_end(ap);
}

static cv::Mat getTensorMat(backscrub_ctx_t &ctx, int tnum) {

	TfLiteType t_type = ctx.interpreter->tensor(tnum)->type;

	if (kTfLiteFloat32 != t_type) {
		_dbg(ctx, "error: tensor #%d: is not float32 type (%d)\n", tnum, t_type);
		return cv::Mat();
	}

	TfLiteIntArray* dims = ctx.interpreter->tensor(tnum)->dims;

	for (int i = 0; i < dims->size; i++)
		_dbg(ctx, "tensor #%d: %d\n", tnum, dims->data[i]);

	if (dims->data[0] != 1) {
		_dbg(ctx, "error: tensor #%d: is not single vector (%d)\n", tnum, dims->data[0]);
		return cv::Mat();
	}

	int h = dims->data[1];
	int w = dims->data[2];
	int c = dims->data[3];

	float* p_data = ctx.interpreter->typed_tensor<float>(tnum);

	if (nullptr == p_data) {
		_dbg(ctx, "error: tensor #%d: unable to obtain data pointer\n", tnum);
		return cv::Mat();
	}

	return cv::Mat(h, w, CV_32FC(c), p_data);
}

// Determine type of model from the name
// TODO:XXX: use metadata when available
static modeltype_t get_modeltype(const std::string& modelname) {
	if (modelname.find("body-pix") != modelname.npos) {
		return modeltype_t::BodyPix;
	}
	else if (modelname.find("deeplab") != modelname.npos) {
		return modeltype_t::DeepLab;
	}
	else if (modelname.find("segm_") != modelname.npos) {
		return modeltype_t::GoogleMeetSegmentation;
	}
	else if (modelname.find("selfie") != modelname.npos) {
		return modeltype_t::MLKitSelfie;
	}

	return modeltype_t::Unknown;
}

static normalization_t get_normalization(modeltype_t type) {
	// TODO: This should be read out from actual model metadata instead
	normalization_t rv = {0};

	switch (type) {
		case modeltype_t::DeepLab:
			rv.scaling = 1 / 127.5;
			rv.offset = -1;
			break;

		case modeltype_t::BodyPix:
		case modeltype_t::GoogleMeetSegmentation:
		case modeltype_t::MLKitSelfie:
		case modeltype_t::Unknown:
		default:
			rv.scaling = 1 / 255.0;
			rv.offset = 0;
			break;
	}

	return rv;
}

// deeplabv3 classes
// TODO: read from model metadata file
static const std::vector<std::string> labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" };
// label number of "person" for DeepLab v3+ model
static const size_t cnum = labels.size();
static const size_t pers = std::distance(labels.begin(), std::find(labels.begin(), labels.end(), "person"));

void *bs_maskgen_new(
	// Required parameters
	const std::string& modelname,
	size_t threads,
	size_t width,
	size_t height,

	// Optional (nullable) callbacks with caller-provided context
	// ..debug output
	void (*ondebug)(void *ctx, const char *msg),
	// ..after preparing video frame
	void (*onprep)(void *ctx),
	// ..after running inference
	void (*oninfer)(void *ctx),
	// ..after generating mask
	void (*onmask)(void *ctx),
	// ..the returned context
	void *caller_ctx
) {
	// Allocate context
	backscrub_ctx_t *pctx = new backscrub_ctx_t;

	// Take a reference so we can write tidy code with ctx.<x>
	backscrub_ctx_t &ctx = *pctx;

	// Save callbacks
	ctx.ondebug = ondebug;
	ctx.onprep = onprep;
	ctx.oninfer = oninfer;
	ctx.onmask = onmask;
	ctx.caller_ctx = caller_ctx;

	// Load model
	ctx.model = tflite::FlatBufferModel::BuildFromFile(modelname.c_str());

	if (!ctx.model) {
		_dbg(ctx, "error: unable to load model from file: '%s'.\n", modelname.c_str());
		bs_maskgen_delete(pctx);
		return nullptr;
	}

	// Determine model type and normalization values
	ctx.modeltype = get_modeltype(modelname);

	if (modeltype_t::Unknown == ctx.modeltype) {
		_dbg(ctx, "error: unknown model type '%s'.\n", modelname.c_str());
		bs_maskgen_delete(pctx);
		return nullptr;
	}

	ctx.norm = get_normalization(ctx.modeltype);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;

	// custom op for Google Meet network
	resolver.AddCustom(
		"Convolution2DTransposeBias",
		mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()
	);
	tflite::InterpreterBuilder builder(*ctx.model, resolver);
	builder(&ctx.interpreter);

	if (!ctx.interpreter) {
		_dbg(ctx, "error: unable to build model interpreter\n");
		bs_maskgen_delete(pctx);
		return nullptr;
	}

	// Allocate tensor buffers.
	if (ctx.interpreter->AllocateTensors() != kTfLiteOk) {
		_dbg(ctx, "error: unable to allocate tensor buffers\n");
		bs_maskgen_delete(pctx);
		return nullptr;
	}

	// set interpreter params
	ctx.interpreter->SetNumThreads(threads);
	ctx.interpreter->SetAllowFp16PrecisionForFp32(true);

	// get input and output tensor as cv::Mat
	ctx.input = getTensorMat(ctx, ctx.interpreter->inputs()[0]);
	ctx.output = getTensorMat(ctx, ctx.interpreter->outputs()[0]);

	if (ctx.input.empty() || ctx.output.empty()) {
		bs_maskgen_delete(pctx);
		return nullptr;
	}

	ctx.img_dim = cv::Rect(0, 0, ctx.input.cols, ctx.input.rows);

	ctx.src_ratio = (float)height / (float)width;
	ctx.net_ratio = (float)ctx.input.rows / (float)ctx.input.cols;

	const auto size_src = backscrub_point_t{width, height};
	const auto size_net = backscrub_point_t(ctx.input.cols, ctx.input.rows);

	auto size_filter = size_net;

	/**
	 * The following code assumes that the source image is larger
	 * than the input for the neuronal network.
	 * If src.x * net.y > src.y * net.x we know that the image has a wider aspect ratio then the network.
	 * If src.x * net.y < src.y * net.x we know that the network has the wider aspect ratio.
	 * In each case we chose the largest rectangle within the source image that fits within the network.
	 * This rectangle is than applied multiple times by sliding it across the source image until all of the source is covered.
	 * When sliding the network window across the source it is ensured that we do an odd number of passes.
	 * This forces at least one window to cover the center region of the image.
	 */

	auto wnd_count = backscrub_point_t{1, 1};

	if (size_src.x * size_net.y > size_src.y * size_net.x) {
		size_filter.x = size_net.x * size_src.y / size_net.y;
		size_filter.y = size_src.y;
		wnd_count.x = 1 | ((size_src.x + size_filter.x - 1) / size_filter.x);
		wnd_count.y = 1;
	} else {
		size_filter.x = size_src.x;
		size_filter.y = size_net.y * size_src.x / size_net.x;
		wnd_count.x = 1;
		wnd_count.y = 1 | ((size_src.y + size_filter.y - 1) / size_filter.y);
	}

	// initialize mask and model-aspect ROI in center
	if (ctx.src_ratio < ctx.net_ratio) {
		// if frame is wider than model, then use only the frame center
		ctx.src_roidim = cv::Rect((width - height / ctx.net_ratio) / 2, 0, height / ctx.net_ratio, height);
		ctx.net_roidim = cv::Rect(0, 0, ctx.input.cols, ctx.input.rows);
	} else {
		// if model is wider than the frame, center the frame in the model
		ctx.src_roidim = cv::Rect(0, 0, width, height);
		ctx.net_roidim = cv::Rect((ctx.input.cols - ctx.input.rows / ctx.src_ratio) / 2, 0, ctx.input.rows / ctx.src_ratio, ctx.input.rows);
	}

	// Item 0 is always a central cut from the image
	ctx.region_rects.clear();
	ctx.region_rects.emplace_back(backscrub_rect_t(
		ctx.src_roidim, ctx.net_roidim
	));

	for(size_t idx_x = 0; idx_x < wnd_count.x; idx_x++) {
		for(size_t idx_y = 0; idx_y < wnd_count.y; idx_x++) {
			const size_t sx = wnd_count.x > 1 ? wnd_count.x - 1 : 1;
			const size_t sy = wnd_count.y > 1 ? wnd_count.y - 1 : 1;

			size_t dx = size_src.x - size_net.x;
			size_t dy = size_src.y - size_net.y;

			dx *= idx_x;
			dy *= idx_y;

			dx /= sx;
			dy /= sy;

			auto src_rect = cv::Rect(dx, dy, size_filter.x, size_filter.y);
			auto dst_rect = cv::Rect(0, 0, ctx.input.cols, ctx.input.rows);

			ctx.region_rects.emplace_back(src_rect, dst_rect);
		}
	}

	ctx.in_u8_bgr = cv::Mat(ctx.input.rows, ctx.input.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	// mask blurring size
	ctx.blur = cv::Size(5, 5);

	// create Mat for small mask
	ctx.ofinal = cv::Mat(ctx.output.rows, ctx.output.cols, CV_8UC1);
	return pctx;
}

void bs_maskgen_delete(void *context) {
	if (!context)
		return;

	backscrub_ctx_t &ctx = *((backscrub_ctx_t *)context);

	// drop interpreter (if present)
	if (ctx.interpreter != nullptr)
		ctx.interpreter.reset();

	// drop model (if present)
	if (ctx.model != nullptr)
		ctx.model.reset();

	delete &ctx;
}

bool bs_maskgen_process(void *context, cv::Mat &frame, cv::Mat &mask) {
	if (!context)
		return false;

	backscrub_ctx_t &ctx = *((backscrub_ctx_t *)context);

	ctx.mask = cv::Mat::ones(ctx.img_dim.height, ctx.img_dim.width, CV_8UC1) * 255;

	for(auto& region: ctx.region_rects) {
		ctx.src_roidim = region.src;
		ctx.net_roidim = region.dst;

		ctx.mask_region = ctx.mask(ctx.src_roidim);

		// map ROI
		cv::Mat roi = frame(ctx.src_roidim);

		cv::Mat in_roi = ctx.in_u8_bgr(ctx.net_roidim);
		cv::resize(roi, in_roi, ctx.net_roidim.size());

		cv::Mat in_u8_rgb;
		cv::cvtColor(ctx.in_u8_bgr, in_u8_rgb, cv::COLOR_BGR2RGB);

		// TODO: can convert directly to float?

		// bilateral filter to reduce noise
		if (1) {
			cv::Mat filtered;
			cv::bilateralFilter(in_u8_rgb, filtered, 5, 100.0, 100.0);
			in_u8_rgb = filtered;
		}

		// convert to float and normalize values expected by the model
		in_u8_rgb.convertTo(ctx.input, CV_32FC3, ctx.norm.scaling, ctx.norm.offset);

		if (ctx.onprep)
			ctx.onprep(ctx.caller_ctx);

		// Run inference
		if (ctx.interpreter->Invoke() != kTfLiteOk) {
			_dbg(ctx, "error: failed to interpret video frame\n");
			return false;
		}

		if (ctx.oninfer)
			ctx.oninfer(ctx.caller_ctx);

		float* tmp = (float*)ctx.output.data;
		uint8_t* out = (uint8_t*)ctx.ofinal.data;

		switch (ctx.modeltype) {
			case modeltype_t::DeepLab:
				// find class with maximum probability
				for (unsigned int n = 0; n < ctx.output.total(); n++) {
					float maxval = -10000;
					size_t maxpos = 0;

					for (size_t i = 0; i < cnum; i++) {
						if (tmp[n * cnum + i] > maxval) {
							maxval = tmp[n * cnum + i];
							maxpos = i;
						}
					}

					// set mask to 0 where class == person
					uint8_t val = (maxpos == pers ? 0 : 255);
					out[n] = (val & 0xE0) | (out[n] >> 3);
				}

				break;

			case modeltype_t::BodyPix:
			case modeltype_t::MLKitSelfie:

				// threshold probability
				for (unsigned int n = 0; n < ctx.output.total(); n++) {
					// FIXME: hardcoded threshold
					uint8_t val = (tmp[n] > 0.65 ? 0 : 255);
					out[n] = (val & 0xE0) | (out[n] >> 3);
				}

				break;

			case modeltype_t::GoogleMeetSegmentation:

				/* 256 x 144 x 2 tensor for the full model or 160 x 96 x 2
				* tensor for the light model with masks for background
				* (channel 0) and person (channel 1) where values are in
				* range [MIN_FLOAT, MAX_FLOAT] and user has to apply
				* softmax across both channels to yield foreground
				* probability in [0.0, 1.0].
				*/
				for (unsigned int n = 0; n < ctx.output.total(); n++) {
					float exp0 = expf(tmp[2 * n    ]);
					float exp1 = expf(tmp[2 * n + 1]);
					float p0 = exp0 / (exp0 + exp1);
					float p1 = exp1 / (exp0 + exp1);
					uint8_t val = (p0 < p1 ? 0 : 255);
					out[n] = (val & 0xE0) | (out[n] >> 3);
				}

				break;

			case modeltype_t::Unknown:
				_dbg(ctx, "error: unknown model type (%d)\n", ctx.modeltype);
				return false;
		}

		if (ctx.onmask)
			ctx.onmask(ctx.caller_ctx);

		// scale up into full-sized mask
		cv::Mat tmpbuf;
		cv::resize(ctx.ofinal(ctx.net_roidim), tmpbuf, ctx.mask_region.size());

		// blur at full size for maximum smoothness
		cv::blur(tmpbuf, ctx.mask_region, ctx.blur);

		// copy out
		mask = ctx.mask;
	}

	return true;
}

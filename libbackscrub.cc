/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

// tested against tensorflow lite v2.4.1 (static library)

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "transpose_conv_bias.h"
#include "libbackscrub.h"

// Tensorflow Lite helper functions
using namespace tflite;

typedef struct {
	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<Interpreter> interpreter;
} backscrub_ctx_t;

static cv::Mat getTensorMat(backscrub_ctx_t &ctx, int tnum, int debug) {

	TfLiteType t_type = ctx.interpreter->tensor(tnum)->type;
	TFLITE_MINIMAL_CHECK(t_type == kTfLiteFloat32);

	TfLiteIntArray* dims = ctx.interpreter->tensor(tnum)->dims;
	if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
	TFLITE_MINIMAL_CHECK(dims->data[0] == 1);

	int h = dims->data[1];
	int w = dims->data[2];
	int c = dims->data[3];

	float* p_data = ctx.interpreter->typed_tensor<float>(tnum);
	TFLITE_MINIMAL_CHECK(p_data != nullptr);

	return cv::Mat(h,w,CV_32FC(c),p_data);
}

// deeplabv3 classes
static const std::vector<std::string> labels = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" };
// label number of "person" for DeepLab v3+ model
static const size_t cnum = labels.size();
static const size_t pers = std::distance(labels.begin(), std::find(labels.begin(),labels.end(),"person"));

int init_tensorflow(calcinfo_t &info) {
	// Allocate context
	info.backscrub_ctx = new backscrub_ctx_t;
	// Take a reference so we can write tidy code with ctx.<x>
	backscrub_ctx_t &ctx = *((backscrub_ctx_t *)info.backscrub_ctx);
	// Load model
	ctx.model = tflite::FlatBufferModel::BuildFromFile(info.modelname);
	TFLITE_MINIMAL_CHECK(ctx.model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	// custom op for Google Meet network
	resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());
	InterpreterBuilder builder(*ctx.model, resolver);
	builder(&ctx.interpreter);
	TFLITE_MINIMAL_CHECK(ctx.interpreter != nullptr);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(ctx.interpreter->AllocateTensors() == kTfLiteOk);

	// set interpreter params
	ctx.interpreter->SetNumThreads(info.threads);
	ctx.interpreter->SetAllowFp16PrecisionForFp32(true);

	// get input and output tensor as cv::Mat
	info.input = getTensorMat(ctx, ctx.interpreter->inputs ()[0],info.debug);
	info.output = getTensorMat(ctx, ctx.interpreter->outputs()[0],info.debug);
	info.ratio = (float)info.input.cols/(float) info.input.rows;

	// initialize mask and square ROI in center
	info.roidim = cv::Rect((info.width-info.height/info.ratio)/2,0,info.height/info.ratio,info.height);
	info.mask = cv::Mat::ones(info.height,info.width,CV_8UC1)*255;
	info.mroi = info.mask(info.roidim);

	// mask blurring size
	info.blur = cv::Size(5,5);

	// create Mat for small mask
	info.ofinal = cv::Mat(info.output.rows,info.output.cols,CV_8UC1);
	return 1;
}

int calc_mask(calcinfo_t &info) {
	// Ensure we have a context from init_tensorflow()
	TFLITE_MINIMAL_CHECK(info.backscrub_ctx!=NULL);
	backscrub_ctx_t &ctx = *((backscrub_ctx_t *)info.backscrub_ctx);

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
	if (info.onprep)
		info.onprep(info.caller_ctx);

	// Run inference
	TFLITE_MINIMAL_CHECK(ctx.interpreter->Invoke() == kTfLiteOk);
	if (info.oninfer)
		info.oninfer(info.caller_ctx);

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
	if (strstr(info.modelname,"body-pix") || strstr(info.modelname,"selfie")) {
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
	if (info.onmask)
		info.onmask(info.caller_ctx);

	// scale up into full-sized mask
	cv::Mat tmpbuf;
	cv::resize(info.ofinal,tmpbuf,cv::Size(info.raw.rows/info.ratio,info.raw.rows));

	// blur at full size for maximum smoothness
	cv::blur(tmpbuf,info.mroi,info.blur);
	return 1;
}


/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

// tested against tensorflow lite v2.4.1 (static library)
#ifndef _LIBBACKSCRUB_H
#define _LIBBACKSCRUB_H

// for cv::Mat and related types
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
	fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

// Ensure we are usable by plain C programs or ffi language bindings
#ifdef __cplusplus
extern "C" {
#endif

// Shared state structure between caller and libbackscrub
typedef struct {
	// Required to be set before calling init_tensorflow
	// XXX:TODO: poss should be formal params - PAA
	const char *modelname;
	size_t threads;
	size_t width;
	size_t height;
	int debug;
	// Optional callbacks with context between processing steps
	void (*onprep)(void *);
	void (*oninfer)(void *);
	void (*onmask)(void *);
	void *caller_ctx;
	// Used by libbackscrub / callbacks (eg: adjusting blur size)
	// XXX:TODO: probably too much coupling - PAA
	cv::Mat input;
	cv::Mat output;
	cv::Rect roidim;
	cv::Mat mask;
	cv::Mat mroi;
	cv::Mat raw;
	cv::Mat ofinal;
	cv::Size blur;
	float ratio;
	void *backscrub_ctx;	// opaque context used by libbackscrub
} calcinfo_t;

extern int init_tensorflow(calcinfo_t &info);
extern int calc_mask(calcinfo_t &info);

#ifdef __cplusplus
}
#endif


#endif

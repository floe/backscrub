/* This is licenced software, @see LICENSE file.
 * Authors - @see AUTHORS file.
==============================================================================*/

#ifndef _LIBBACKSCRUB_H
#define _LIBBACKSCRUB_H

// for cv::Mat and related types
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Return a new (opaque) mask generation context
extern void *bs_maskgen_new(
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
);

// Delete the mask generation context
extern void bs_maskgen_delete(void *context);

// Process a video frame into a mask
extern bool bs_maskgen_process(void *context, cv::Mat& frame, cv::Mat &mask);

#endif

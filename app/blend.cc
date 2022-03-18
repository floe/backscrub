#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "utils.h"


cv::Mat alpha_blend(const cv::Mat& srca, const cv::Mat& srcb, const cv::Mat& mask) {
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

	const uint8_t *aptr = (const uint8_t*)srca.data;
	const uint8_t *bptr = (const uint8_t*)srcb.data;
	const uint8_t *mptr = (const uint8_t*)mask.data;

	cv::Mat out;
	out.create(srca.size(), srca.type());
	uint8_t *optr = (uint8_t*)out.data;

	// by removing this to a constant, and using const weights, GCC can vectorize this loop
	const size_t npix = srca.rows * srca.cols;
	for (size_t pix = 0; pix < npix; ++pix) {
		// calculate pre-multipied weights
		const uint32_t aw = (uint32_t)(*mptr++)*257;
		const uint32_t bw = 65535 - aw;
		// blend each channel byte
		*optr++ = (uint8_t)(( (uint32_t)(*aptr++) * aw + (uint32_t)(*bptr++) * bw ) >> 16);
		*optr++ = (uint8_t)(( (uint32_t)(*aptr++) * aw + (uint32_t)(*bptr++) * bw ) >> 16);
		*optr++ = (uint8_t)(( (uint32_t)(*aptr++) * aw + (uint32_t)(*bptr++) * bw ) >> 16);
	}
	return out;
}

cv::Mat convert_rgb_to_yuyv(const cv::Mat& input) {
	cv::Mat tmp;
	cv::cvtColor(input, tmp, cv::COLOR_RGB2YUV);
	std::vector<cv::Mat> yuv;
	cv::split(tmp, yuv);
	cv::Mat yuyv(tmp.rows, tmp.cols, CV_8UC2);

	uint8_t* outdata = (uint8_t*)yuyv.data;
	uint8_t* ydata = (uint8_t*)yuv[0].data;
	uint8_t* udata = (uint8_t*)yuv[1].data;
	uint8_t* vdata = (uint8_t*)yuv[2].data;

	// removing this to a const, and using const u/v values, GCC can vectorize this loop
	const size_t total = yuyv.total();
	for (size_t i = 0; i < total; i += 2) {
		const uint8_t u = (uint8_t)(((int)udata[i] + (int)udata[i + 1]) / 2);
		const uint8_t v = (uint8_t)(((int)vdata[i] + (int)vdata[i + 1]) / 2);

		outdata[2 * i + 0] = ydata[i + 0];
		outdata[2 * i + 1] = v;
		outdata[2 * i + 2] = ydata[i + 1];
		outdata[2 * i + 3] = u;
	}

	return yuyv;
}

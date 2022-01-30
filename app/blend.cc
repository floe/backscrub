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

	for (size_t pix = 0, npix = srca.rows * srca.cols; pix < npix; ++pix) {
		// blending weights
		int aw = (int)(*mptr++);
		int bw = 255 - aw;

		// blend each channel byte
		*optr++ = (uint8_t)(( (int)(*aptr++) * aw + (int)(*bptr++) * bw ) >> 8);
		*optr++ = (uint8_t)(( (int)(*aptr++) * aw + (int)(*bptr++) * bw ) >> 8);
		*optr++ = (uint8_t)(( (int)(*aptr++) * aw + (int)(*bptr++) * bw ) >> 8);
	}
	return out;
}

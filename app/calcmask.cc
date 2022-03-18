#include "calcmask.h"

#include <cstddef>
#include <cstdio>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#include "app/utils.h"

#include "lib/libbackscrub.h"


CalcMask::CalcMask(
    const std::string& modelname,
    size_t threads,
    size_t width,
    size_t height
) {
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

	// Start the actual processing
	state = thread_state_t::RUNNING;
	thread = std::thread(&CalcMask::run, this);
}

CalcMask::~CalcMask() {
	// mark as done
	state = thread_state_t::DONE;

	// wake up processing thread
	new_frame = true;
	condition_new_frame.notify_all();

	// collect termination
	thread.join();

	// free resources
	bs_maskgen_delete(maskctx);
}

void CalcMask::set_input_frame(cv::Mat &frame) {
	std::lock_guard<std::mutex> hold(lock_frame);

	*frame_next = frame.clone();
	new_frame = true;
	condition_new_frame.notify_all();
}

void CalcMask::get_output_mask(cv::Mat &out) {
	if (new_mask) {
		std::lock_guard<std::mutex> hold(lock_mask);

		out = mask_out->clone();
		new_mask = false;
	}
}

void CalcMask::run() {
	cv::Mat *raw_tmp;
	timestamp_t tloop;

	while(thread_state_t::RUNNING == this->state) {
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
			this->state = thread_state_t::DONE;
			return;
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
void CalcMask::onprep(void *ctx) {
	CalcMask *cls = (CalcMask *)ctx;
	cls->prepns = diffnanosecs(timestamp(), cls->t0);
	cls->t0 = timestamp();
}

void CalcMask::oninfer(void *ctx) {
	CalcMask *cls = (CalcMask *)ctx;
	cls->tfltns = diffnanosecs(timestamp(), cls->t0);
	cls->t0 = timestamp();
}

void CalcMask::onmask(void *ctx) {
	CalcMask *cls = (CalcMask *)ctx;
	cls->maskns = diffnanosecs(timestamp(), cls->t0);
	cls->t0 = timestamp();
}

#pragma once

#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core/mat.hpp>

#include "utils.h"

enum class thread_state_t { RUNNING, DONE };

class CalcMask final {
protected:
	volatile thread_state_t state;

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

	// thread execution
	void run();

	// timing callbacks
	static void onprep(void *ctx);
	static void oninfer(void *ctx);
	static void onmask(void *ctx);

public:
	long waitns;
	long prepns;
	long tfltns;
	long maskns;
	long loopns;

	CalcMask() = delete;
	CalcMask(const CalcMask&) = delete;
	CalcMask(const std::string& modelname, size_t threads, size_t width, size_t height);
	~CalcMask();

	void set_input_frame(cv::Mat &frame);
	void get_output_mask(cv::Mat &out);
};

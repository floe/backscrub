#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>


// Due to weirdness in the C(++) preprocessor, we have to nest stringizing macros to ensure expansion
// http://gcc.gnu.org/onlinedocs/cpp/Stringizing.html, use _STR(<raw text or macro>).
#define __STR(X) #X
#define _STR(X) __STR(X)

// timing helpers
typedef std::chrono::high_resolution_clock::time_point timestamp_t;
extern timestamp_t timestamp();
extern long diffnanosecs(const timestamp_t& t1, const timestamp_t& t2);

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

extern int fourCcFromString(const std::string& in);

extern cv::Mat convert_rgb_to_yuyv(const cv::Mat& input);

extern cv::Mat alpha_blend(const cv::Mat& srca, const cv::Mat& srcb, const cv::Mat& mask);

extern bool is_number(const std::string &s);

extern std::optional<std::string> resolve_path(const std::string& provided, const std::string& type);

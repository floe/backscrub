/* This is licensed software, @see LICENSE file.
 * Authors - @see AUTHORS file. */

#ifndef _BACKGROUND_H_
#define _BACKGROUND_H_

#include <opencv2/core/mat.hpp>

struct background_t;

// Load  a background media path (image or video file, network stream [URL])
// Returns opaque handle or nullptr on error. The returned shared_ptr will
// clean up after itself during deletion
std::shared_ptr<background_t> load_background(const std::string& path, int debug);

// Grab current frame from background
// Returns current frame number (1 for still image) or -1 on error
// NB: current frame can loop round to 0!
int grab_background(std::shared_ptr<background_t> handle, int width, int height, cv::Mat &out);

// Grab current thumbnail image (if any) from background
// Returns <0 on error, 0 on success and copies thumbnail to out
int grab_thumbnail(std::shared_ptr<background_t> handle, cv::Mat &out);

#endif

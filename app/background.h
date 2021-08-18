/* This is licensed software, @see LICENSE file.
 * Authors - @see AUTHORS file. */

#ifndef _BACKGROUND_H_
#define _BACKGROUND_H_

#include <opencv2/core/mat.hpp>

// Load  a background media path (image or video file, network stream [URL])
// Returns opaque handle or nullptr on error
void *load_background(const char *path, int debug);

// Grab current frame from background
// Returns current frame number (1 for still image) or -1 on error
// NB: current frame can loop round to 0!
int grab_background(void *handle, int width, int height, cv::Mat &out);

// Release background handle (if any)
void drop_background(void *handle);

#endif

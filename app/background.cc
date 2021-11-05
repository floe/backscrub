/* This is licensed software, @see LICENSE file.
 * Authors - @see AUTHORS file. */

#include <stdio.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Internal state of background processing
struct background_t {
    int debug;
    bool video;
    volatile bool run;
    cv::VideoCapture cap;
    int frame;
    double fps;
    cv::Mat raw;
    cv::Mat thumb;
    std::thread thread;
    std::mutex rawmux;
    std::mutex thumbmux;
};

// Internal video reader thread
static void read_thread(std::shared_ptr<background_t> pbkd) {
    if (pbkd->debug) fprintf(stderr, "background: thread start\n");
    auto last = std::chrono::steady_clock::now();
    auto next = last;
    while (pbkd->run) {
        // read new frame - we use a temporary buffer for two reasons:
        // - we can read unlocked, thus we are not impacted by blocking backends (eg: V4L2)
        // - we preserve the last frame for callers, such that if a video ends, they always have a frame to use
        cv::Mat grab;
        if (pbkd->cap.read(grab)) {
            // got a frame, lock and copy for callers
            {
                std::unique_lock<std::mutex> hold(pbkd->rawmux);
                grab.copyTo(pbkd->raw);
                pbkd->frame += 1;
            }
            // grab timing point
            auto now = std::chrono::steady_clock::now();
            // generate thumbnail frame with overlay info if double debug enabled
            if (pbkd->debug > 1) {
                char msg[40];
                long nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count();
                double fps = 1e9/(double)nsec;
                {
                    std::unique_lock<std::mutex> hold(pbkd->thumbmux);
                    cv::resize(grab, pbkd->thumb, cv::Size(160,120));
                    snprintf(msg, sizeof(msg), "FPS:%0.1f", fps);
                    cv::putText(pbkd->thumb, msg, cv::Point(5,15), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255));
                    snprintf(msg, sizeof(msg), "FRM:%05d", fps, pbkd->frame);
                    cv::putText(pbkd->thumb, msg, cv::Point(5,30), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255));
                    cv::putText(pbkd->thumb, "Background", cv::Point(5,pbkd->thumb.rows-5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255));
                }
            }
            last = now;
            // wait for next frame, some sources are real-time, others are not, this ensures all play in real-time.
            next += std::chrono::nanoseconds((long)(1e9/pbkd->fps));
            while (now < next) {
                std::this_thread::sleep_until(next);
                now = std::chrono::steady_clock::now();
            }
        } else {
            // no more frames, but if we processed some, try to reset position and go again
            if (pbkd->frame>0 && pbkd->cap.set(cv::CAP_PROP_POS_FRAMES, 0)) {
                std::unique_lock<std::mutex> hold(pbkd->rawmux);
                pbkd->frame = 0;
            } else {
                // unable to reset or previous attempt produced no more frames - stop
                if (pbkd->debug) fprintf(stderr, "background: thread stopping at end of stream and not resettable\n");
                break;
            }
        }
    }
    if (pbkd->debug) fprintf(stderr, "background: thread stop\n");
}

static void drop_background(background_t *pbkd) {
    if (!pbkd)
        return;
    if (pbkd->video) {
        // stop capture
        if (pbkd->run) {
            pbkd->run = false;
            pbkd->thread.join();
        }
        // clean up
        pbkd->cap.release();
        pbkd->raw.release();
        pbkd->thumb.release();
    } else {
        // clean up
        pbkd->raw.release();
    }
    delete pbkd;
}

std::shared_ptr<background_t> load_background(const std::string& path, int debug) {
    // allocate a shared pointer around storage for the handle, associate custom deleter to clean up when eventually released
    auto pbkd = std::shared_ptr<background_t>(new background_t, drop_background);
    try {
        pbkd->debug = debug;
        pbkd->video = false;
        pbkd->run = false;
        pbkd->cap.open(path, cv::CAP_ANY);    // explicitly ask for auto-detection of backend
        if (!pbkd->cap.isOpened()) {
            if (pbkd->debug) fprintf(stderr, "background: cap cannot open: %s\n", path.c_str());
            return nullptr;
        }
        pbkd->cap.set(cv::CAP_PROP_CONVERT_RGB, true);
        int fcc = (int)pbkd->cap.get(cv::CAP_PROP_FOURCC);
        pbkd->fps = pbkd->cap.get(cv::CAP_PROP_FPS);
        int cnt = (int)pbkd->cap.get(cv::CAP_PROP_FRAME_COUNT);
        // Here be the logic...
        //  if: can read 2 video frames => it's a video
        //  else: is loaded as an image => it's an image
        //  else: it's not usable.
        if (pbkd->cap.read(pbkd->raw) && pbkd->cap.read(pbkd->raw)) {
            // it's a video, try a reset and start reader thread..
            if (pbkd->cap.set(cv::CAP_PROP_POS_FRAMES, 0))
                pbkd->frame = 0;
            else
                pbkd->frame = 2;    // unable to reset, so we're 2 frames in
            pbkd->video = true;
            pbkd->run = true;
            pbkd->thread = std::thread(read_thread, pbkd);
        } else {
            // static image file, try loading..
            pbkd->cap.release();
            pbkd->raw = cv::imread(path);
            if (pbkd->raw.empty()) {
                if (pbkd->debug) fprintf(stderr, "background: imread cannot open: %s\n", path.c_str());
                return nullptr;
            }
        }
        if (pbkd->debug)
            fprintf(stderr, "background properties:\n\tvid: %s\n\tfcc: %08x (%.4s)\n\tfps: %f\n\tcnt: %d\n",
                pbkd->video ? "yes":"no", fcc, (char *)&fcc, pbkd->fps, cnt);
    } catch (std::exception const &e) {
        // oops
        if (pbkd->debug) fprintf(stderr, "background: exception while loading: %s\n", e.what());
        return nullptr;
    } catch (...) {
        if (pbkd->debug) fprintf(stderr, "background: unknown exception\n");
        return nullptr;
    }
    return pbkd;
}

int grab_background(std::shared_ptr<background_t> pbkd, int width, int height, cv::Mat& out) {
    if (!pbkd)
        return -1;
    // static image or video?
    int frm ;
    if (pbkd->video) {
        // grab frame & frame no. under mutex
        std::unique_lock<std::mutex> hold(pbkd->rawmux);
        cv::resize(pbkd->raw, out, cv::Size(width, height));
        frm = pbkd->frame;
    } else {
        // resize still image as requested into out
        cv::resize(pbkd->raw, out, cv::Size(width, height));
        frm = 1;
    }
    return frm;
}

int grab_thumbnail(std::shared_ptr<background_t> pbkd, cv::Mat &out) {
    if (!pbkd)
        return -1;
    std::unique_lock<std::mutex> hold(pbkd->thumbmux);
    out = pbkd->thumb.clone();
    return 0;
}

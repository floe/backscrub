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
    int dbg;
    bool vid;
    volatile bool run;
    cv::VideoCapture cap;
    int frm;
    double fps;
    cv::Mat raw;
    std::thread thr;
    std::mutex mux;
};

// Internal video reader thread
static void read_thread(std::shared_ptr<background_t> pbkd) {
    if (pbkd->dbg) fprintf(stderr, "background: thread start\n");
    auto last = std::chrono::steady_clock::now();
    auto proc = last;
    while (pbkd->run) {
        // read new frame - we use a temporary buffer for two reasons:
        // - we can read unlocked, thus we are not impacted by blocking backends (eg: V4L2)
        // - we preserve the last frame for callers, such that if a video ends, they always have a frame to use
        cv::Mat grab;
        if (pbkd->cap.read(grab)) {
            // got a frame, lock and copy for callers
            {
                std::unique_lock<std::mutex> hold(pbkd->mux);
                grab.copyTo(pbkd->raw);
                pbkd->frm += 1;
            }
            // grab timing point
            auto now = std::chrono::steady_clock::now();
            // display thumbnail frame with overlay info if debug enabled
            if (pbkd->dbg) {
                char msg[40];
                long nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count();
                snprintf(msg, sizeof(msg), "FPS:%0.1f FRM:%d", 1e9/(double)nsec, pbkd->frm);
                cv::Mat frame;
                cv::resize(grab, frame, cv::Size(240,160));
                cv::putText(frame, msg, cv::Point(5,frame.rows-5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
                cv::imshow("Background", frame);
            }
            last = now;
            // wait for next frame, some sources are real-time, others are not, this ensures all play in real-time.
            if (pbkd->frm>1) {
                auto adj = std::chrono::duration_cast<std::chrono::nanoseconds>(now-proc);
                auto del = std::chrono::nanoseconds((long)(1e9/pbkd->fps));
                // adjust for processing time and sleep or skip if no time left
                if (del > adj) {
                    del -= adj;
                    std::this_thread::sleep_for(del);
                }
                proc = std::chrono::steady_clock::now();
            }
        } else {
            // no more frames, but if we processed some, try to reset position and go again
            if (pbkd->frm>0 && pbkd->cap.set(cv::CAP_PROP_POS_FRAMES, 0)) {
                std::unique_lock<std::mutex> hold(pbkd->mux);
                pbkd->frm = 0;
            } else {
                // unable to reset or previous attempt produced no more frames - stop
                if (pbkd->dbg) fprintf(stderr, "background: thread stopping at end of stream and not resettable\n");
                break;
            }
        }
    }
    if (pbkd->dbg) fprintf(stderr, "background: thread stop\n");
}

std::shared_ptr<background_t> load_background(const char *path, int debug) {
    std::shared_ptr<background_t> pbkd(new background_t);
    try {
        pbkd->dbg = debug;
        pbkd->cap.open(path, cv::CAP_ANY);    // explicitly ask for auto-detection of backend
        if (!pbkd->cap.isOpened()) {
            if (pbkd->dbg) fprintf(stderr, "background: cap cannot open: %s\n", path);
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
                pbkd->frm = 0;
            else
                pbkd->frm = 2;    // unable to reset, so we're 2 frames in
            pbkd->vid = true;
            pbkd->run = true;
            pbkd->thr = std::thread(read_thread, pbkd);
        } else {
            // static image file, try loading..
            pbkd->cap.release();
            pbkd->raw = cv::imread(path);
            pbkd->vid = false;
            if (pbkd->raw.empty()) {
                if (pbkd->dbg) fprintf(stderr, "background: imread cannot open: %s\n", path);
                return nullptr;
            }
        }
        if (pbkd->dbg) fprintf(stderr, "background properties:\n\tvid: %s\n\tfcc: %08x (%.4s)\n\tfps: %f\n\tcnt: %d\n",
            pbkd->vid ? "yes":"no", fcc, (char *)&fcc, pbkd->fps, cnt);
    } catch (std::exception const &e) {
        // oops
        if (pbkd->dbg) fprintf(stderr, "background: exception while loading: %s\n", e.what());
        return nullptr;
    } catch (...) {
        if (pbkd->dbg) fprintf(stderr, "background: unknown exception\n");
        return nullptr;
    }
    return pbkd;
}

int grab_background(std::shared_ptr<background_t> pbkd, int width, int height, cv::Mat& out) {
    if (!pbkd)
        return -1;
    // static image or video?
    int frm ;
    if (pbkd->vid) {
        // grab frame & frame no. under mutex
        std::unique_lock<std::mutex> hold(pbkd->mux);
        cv::resize(pbkd->raw, out, cv::Size(width, height));
        frm = pbkd->frm;
    } else {
        // resize still image as requested into out
        cv::resize(pbkd->raw, out, cv::Size(width, height));
        frm = 1;
    }
    return frm;
}

void drop_background(std::shared_ptr<background_t> pbkd) {
    if (!pbkd)
        return;
    if (pbkd->vid) {
        // stop capture
        pbkd->run = false;
        pbkd->thr.join();
        // clean up
        pbkd->cap.release();
        pbkd->raw.release();
    } else {
        // clean up
        pbkd->raw.release();
    }
}

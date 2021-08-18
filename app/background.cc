/* This is licensed software, @see LICENSE file.
 * Authors - @see AUTHORS file. */

#include <stdio.h>
#include <time.h>
#include <unistd.h>
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
static void read_thread(background_t *pbkd) {
    if (pbkd->dbg) fprintf(stderr, "background: thread start\n");
    struct timespec last;
    struct timespec proc;
    clock_gettime(CLOCK_REALTIME, &last);
    proc = last;
    while (pbkd->run) {
        // read frame under lock
        pbkd->mux.lock();
        if (pbkd->cap.read(pbkd->raw)) {
            pbkd->frm += 1;
            // grab timing point and calculate frame period
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            double nsec = (double)(now.tv_sec-last.tv_sec)*1e9 + (double)(now.tv_nsec-last.tv_nsec);
            last = now;
            // display thumbnail frame with overlay info if debug enabled
            if (pbkd->dbg) {
                char msg[40];
                snprintf(msg, sizeof(msg), "FPS:%0.1f FRM:%d", 1e9/nsec, pbkd->frm);
                cv::Mat frame;
                cv::resize(pbkd->raw, frame, cv::Size(240,160));
                cv::putText(frame, msg, cv::Point(5,frame.rows-5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
                cv::imshow("Background", frame);
            }
            pbkd->mux.unlock();
            // wait for next frame, some sources are real-time, others are not, this ensures all play in real-time.
            if (pbkd->frm>1) {
                long adj = (now.tv_sec-proc.tv_sec)*1000000000l + (now.tv_nsec-proc.tv_nsec);
                now.tv_sec = 0;
                now.tv_nsec= (long)(1.0/pbkd->fps*1e9);
                // adjust for processing time or skip if less than that
                if (now.tv_nsec > adj) {
                    now.tv_nsec -= adj;
                    nanosleep(&now, nullptr);
                } else {
                    if (pbkd->dbg) fprintf(stderr, "background: wait=%ld adj=%ld\n", now.tv_nsec, adj);
                }
                clock_gettime(CLOCK_REALTIME, &proc);
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

void *load_background(const char *path, int debug) {
    background_t *pbkd = new background_t;
    background_t &bkd = *pbkd;

    bkd.dbg = debug;
    bkd.cap.open(path, cv::CAP_ANY);    // explicitly ask for auto-detection of backend
    if (!bkd.cap.isOpened()) {
        if (bkd.dbg) fprintf(stderr, "cv::VideoCapture cannot open: %s\n", path);
        delete pbkd;
        return nullptr;
    }
    bkd.cap.set(cv::CAP_PROP_CONVERT_RGB, true);
    int fcc = (int)bkd.cap.get(cv::CAP_PROP_FOURCC);
    bkd.fps = bkd.cap.get(cv::CAP_PROP_FPS);
    int cnt = (int)bkd.cap.get(cv::CAP_PROP_FRAME_COUNT);
    // Here be the logic...
    //  if: can read 2 video frames => it's a video
    //  else: is loaded as an image => it's an image
    //  else: it's not usable.
    if (bkd.cap.read(bkd.raw) && bkd.cap.read(bkd.raw)) {
        // it's a video, try a reset and start reader thread..
        if (bkd.cap.set(cv::CAP_PROP_POS_FRAMES, 0))
            bkd.frm = 0;
        else
            bkd.frm = 2;
        bkd.vid = true;
        bkd.run = true;
        bkd.thr = std::thread(read_thread, pbkd);
    } else {
        // static image file, try loading..
        bkd.cap.release();
        bkd.raw = cv::imread(path);
        bkd.vid = false;
        if (bkd.raw.empty()) {
            if (bkd.dbg) fprintf(stderr, "cv::imread cannot read: %s\n", path);
            delete pbkd;
            return nullptr;
        }
    }
    if (bkd.dbg) fprintf(stderr, "background properties:\n\tvid: %s\n\tfcc: %08x (%.4s)\n\tfps: %f\n\tcnt: %d\n",
        bkd.vid ? "yes":"no", fcc, (char *)&fcc, bkd.fps, cnt);
    return pbkd;
}

int grab_background(void *handle, int width, int height, cv::Mat& out) {
    background_t *pbkd = (background_t *)handle;
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

void drop_background(void *handle) {
    if (handle) {
        background_t *pbkd = (background_t *)handle;
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
        delete pbkd;
    }
}

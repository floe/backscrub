# DeepBackSub

## Virtual Video Device for Background Replacement with Deep Semantic Segmentation

![Screenshots with my stupid grinning face](screenshot.jpg)
(Credits for the nice backgrounds to [Mary Sabell](https://dribbble.com/shots/4686178-Bauhaus-Poster) and [PhotoFunia](https://photofunia.com/effects/retro-wave))

In these modern times where everyone is sitting at home and skype-ing/zoom-ing/webrtc-ing all the time, I was a bit annoyed about always showing my messy home office to the world. Skype has a "blur background" feature, but that starts to get boring after a while (and it's less private than I would personally like). Zoom has some background substitution thingy built-in, but I'm not touching that software with a bargepole (and that feature is not available on Linux anyway). So I decided to look into how to roll my own implementation without being dependent on any particular video conferencing software to support this.

This whole shebang involves three main steps with varying difficulty:
  - find person in video (hard)
  - replace background (easy)
  - pipe data to virtual video device (medium)

## Finding person in video

### Attempt 0: Depth camera (Intel Realsense)

I've been working a lot with depth cameras previously, also for background segmentation (see [SurfaceStreams](https://github.com/floe/surface-streams)), so I just grabbed a leftover RealSense camera from the lab and gave it a shot. However, the depth data in a cluttered office environment is quite noisy, and no matter how I tweaked the camera settings, it could not produce any depth data for my hair...? I looked like a medieval monk who had the top of his head chopped off, so ... next.

### Attempt 1: OpenCV BackgroundSubtractor

See https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html for tutorial.
Should work OK for mostly static backgrounds and small moving objects, but does not work for a mostly static person in front of a static background. Next.

### Attempt 2: OpenCV Face Detector

See https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html for tutorial.
Works okay-ish, but obviously only detects the face, and not the rest of the person. Also, only roughly matches an ellipse which is looking rater weird in the end. Next.

### Attempt 3: Deep learning!

I've heard good things about this deep learning stuff, so let's try that. I first had to find my way through a pile of frameworks (Keras, Tensorflow, PyTorch, etc.), but after I found a ready-made model for semantic segmentation based on Tensorflow Lite ([DeepLab v3+](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)), I settled on that.

I had a look at the corresponding [Python example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py), [C++ example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image), and [Android example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android), and based on those, I first cobbled together a [Python demo](https://github.com/floe/deepbacksub/blob/master/deepseg.py). That was running at about 2.5 FPS, which is really excruciatingly slow, so I built a [C++ version](https://github.com/floe/deepbacksub/blob/master/deepseg.cc) which manages 10 FPS without too much hand optimization. Good enough.

## Replace Background

This is basically one line of code with OpenCV: `bg.copyTo(raw,mask);` Told you that's the easy part.

## Virtual Video Device

I'm using [v4l2loopback](https://github.com/umlaeute/v4l2loopback) to pipe the data from my userspace tool into any software that can open a V4L2 device. This isn't too hard because of the nice examples, but there are some catches, most notably color space. It took quite some trial and error to find a common pixel format that's accepted by Firefox, Skype, and guvcview, and that is [YUYV](https://www.linuxtv.org/downloads/v4l-dvb-apis-old/V4L2-PIX-FMT-YUYV.html). Nicely enough, my webcam can output YUYV directly as raw data, so that does save me some colorspace conversions.

## End Result

The dataflow through the whole program is roughly as follows:

  - init
    - load background.png, convert to YUYV
    - load DeepLab v3+ network, initialize TFLite
    - setup V4L2 Loopback device (w,h,YUYV)
  - loop
    - grab raw YUYV image from camera
    - extract square ROI in center
      - downscale ROI to 257 x 257 (*)
      - convert to RGB (*)
      - run DeepLab v3+
      - convert result to binary mask for class "person"
      - denoise mask using erode/dilate
    - upscale mask to raw image size
    - copy background over raw image with mask (see above)
    - `write()` data to virtual video device

(*) these are required input parameters for DeepLab v3+

## Requirements

Tested with the following dependencies:
  - Ubuntu 18.04.5, x86-64
  - Linux kernel 4.15 (stock package)
  - OpenCV 3.2.0 (stock package)
  - V4L2-Loopback 0.10.0 (stock package)
  - Tensorflow Lite 2.1.0 (from [repo](https://github.com/tensorflow/tensorflow/tree/v2.1.0/tensorflow/lite))
    - Ultra-short build guide for Tensorflow Lite C++ library: clone repo above, then...
      - run `./tensorflow/lite/tools/make/download_dependencies.sh`
      - run `./tensorflow/lite/tools/make/build_lib.sh`
  
Tested with the following software:
  - Firefox 
    - 74.0.1 (works)
    - 76.0.1 (works)
  - Skype 
    - 8.58.0.93 (works)
    - 8.60.0.76 (works)
  - guvcview 2.0.5 (works with parameter `-c read`)
  - Microsoft Teams 1.3.00.5153 (works)
  - Chrome 81.0.4044.138 (works)
  - Zoom 5.0.403652.0509 (works - yes, I'm a hypocrite, I tested it with Zoom after all :-)
  
## Usage

First, load the v4l2loopback module (extra settings needed to make Chrome work):
```
sudo modprobe v4l2loopback devices=1 max_buffers=2 exclusive_caps=1 card_label="VirtualCam"
```
Then, run deepseg (-d for debug, -c for capture device, -v for virtual device):
```
./deepseg -d -c /dev/video0 -v /dev/video1
```

## Limitations/Extensions

As usual: pull requests welcome.
  - The project name isn't catchy enough. Help me find a nice [backronym](https://en.wikipedia.org/wiki/Backronym).
  - Resolution is currently hardcoded to 640x480 (lowest common denominator).
  - Only works with Linux, because that's what I use.
  - Needs a webcam that can produce raw YUYV data (but extending to the common YUV420 format should be trivial)
  - CPU hog: maxes out two cores on my 2.7 GHz i5 machine for just VGA @ 10 FPS.
  - Uses stock Deeplab v3+ network. Maybe re-training with only "person" and "background" classes could improve performance?

## Fixed
  
  - Should probably do a erosion (+ dilation?) operation on the mask.
  - Background image size needs to match camera resolution (see issue #1).

## Other links

Firefox preferred formats: https://dxr.mozilla.org/mozilla-central/source/media/webrtc/trunk/webrtc/modules/video_capture/linux/video_capture_linux.cc#142-159

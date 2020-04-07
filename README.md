# DeepSeg

## Virtual Video Device for Background Replacement with Deep Semantic Segmentation

In these modern times where everyone is sitting at home and skype-ing/zoom-ing/webrtc-ing all the time, I was a bit annoyed about always showing my messy home office to the world. Skype has a "blur background" feature, but that's also getting boring after a while. Zoom has something similar built-in, but I'm not touching that software with a bargepole. So I decided to look into how to roll my own implementation without being dependent on any particular video conferencing software to support this.

This whole shebang involves three main steps with varying difficulty:
  - find person in video (hard)
  - replace background (easy)
  - pipe data to virtual video device (medium)

## Finding person in video

### Attempt 1: OpenCV BackgroundSubtractor

See https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html for tutorial.
Should work OK for mostly static backgrounds and small moving objects, but does not work for a mostly static person in front of a static background. Next.

### Attempt 2: OpenCV Face Detector

See https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html for tutorial.
Works okay-ish, but obviously only detects the face, and not the rest of the person. Also, only roughly matches an ellipse which is looking rater weird in the end. Next.

### Attempt 3: Deep learning!

I've heard good things about this deep learning stuff, so let's try that. I first had to find my way through a pile of frameworks (Keras, Tensorflow, PyTorch, etc.), but after I found a ready-made model for semantic segmentation based on Tensorflow Lite ([DeepLab v3+](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1)), I settled on that.

I had a look at the corresponding [Python example](), [C++ example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image), and [Android example](), and based on those, I first cobbled together a [Python demo](). That was running at about 2.5 FPS, which is really excruciatingly slow, so I built a [C++ version]() which manages 10 FPS without too much hand optimization. Good enough.

## Replace Background

This is basically one line of code: `bg.copyTo(raw,mask);` Told you that's the easy part.

## Virtual Video Device

I'm using [v4l2loopback][https://github.com/umlaeute/v4l2loopback] to pipe the data from my userspace tool into any software that can open a V4L2 device. This isn't too hard because of the nice examples, but there are some catches, most notably color space. It took quite some trial and error to find a common pixel format that's accepted by Firefox, Skype, and guvcview, and that is `YUYV`. Nicely enough, my webcam can output YUYV directly as raw data, so that does save me some colorspace conversions.

## End Result

The dataflow through the whole program is roughly as follows:

  - init
    - load background image, convert to YUYV
    - load DeepLab v3+ network, initialize TFLite
    - setup V4L2 Loopback device (w,h,YUYV)
  - loop
    - grab raw YUYV image from camera
    - extract square ROI in center
      - downscale ROI to 257 x 257 (*)
      - convert to RGB (*)
      - run DeepLab v3+
      - convert result to binary mask for class "person"
    - upscale mask to raw image size
    - copy background over raw image with mask (see above)
    - `write(2)` data to virtual video device

(*) these are required input parameters for DeepLab v3+

### Links

Model: https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite

Firefox preferred formats: https://dxr.mozilla.org/mozilla-central/source/media/webrtc/trunk/webrtc/modules/video_capture/linux/video_capture_linux.cc#142-159

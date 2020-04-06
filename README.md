# DeepSeg

## Virtual Background Replacement Video Device using Deep Semantic Segmentation

In these modern times where everyone is sitting at home and skype-ing/zoom-ing/webrtc-ing all the time, I was a bit annoyed about always showing my messy home office to the world. Skype has a "blur background" feature, but that's also getting boring after a while. So I decided to look into how to roll my own implementation without being dependent on any particular video conferencing software to support this.

### Attempt 1: OpenCV BackgroundSubtractor

See https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html for tutorial.
Should work OK for mostly static backgrounds and small moving objects, but does not work for a mostly static person in front of a static background. Next.

### Attempt 2: OpenCV Face Detector

See https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html for tutorial.
Works okay-ish, but obviously only detects the face, and not the rest of the person. Also, only roughly matches an ellipse which is looking weird in the end. Next.

### Attempt 3: deep learning!

### Links

Model: https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
C++ example: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image
Firefox preferred formats: https://dxr.mozilla.org/mozilla-central/source/media/webrtc/trunk/webrtc/modules/video_capture/linux/video_capture_linux.cc#142-159

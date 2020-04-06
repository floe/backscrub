Model: https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
C++ example: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image
Test image: https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp
Firefox preferred formats: https://dxr.mozilla.org/mozilla-central/source/media/webrtc/trunk/webrtc/modules/video_capture/linux/video_capture_linux.cc#142-159

    val labelsArrays = arrayOf(
      "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
      "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
      "person", "potted plant", "sheep", "sofa", "train", "tv"
    )

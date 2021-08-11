#!/usr/bin/python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import tensorflow as tf # TF2

# category labels for deeplabv3_257_mv_gpu.tflite
labels = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" ]

if __name__ == '__main__':

  cap = cv2.VideoCapture(0)
  bg = cv2.imread("bauhaus_nothing.jpg")

  print("using NumPy  version "+np.__version__)
  print("using TFLite version "+tf.__version__)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='deeplabv3_257_mv_gpu.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  args = parser.parse_args()

  interpreter = tf.lite.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  while True:
    e1 = cv2.getTickCount()

    ret, img = cap.read()
    roi = img[0:540,210:750] # row0:row1, col0:col1
    img = cv2.resize(roi,(width,height))
    #img = cv2.resize(cv2.imread(args.image),(width,height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # find the highest-probability class for each pixel (along axis 2)
    out = np.apply_along_axis(np.argmax,2,results)
      
    # set pixels with likeliest class == person to 255
    pers_idx = labels.index("person")
    person = np.where(out == pers_idx, 255, 0).astype(np.uint8)

    # use mask to combine with background
    tmp1 = cv2.bitwise_and(img, img, mask=person)
    tmp2 = cv2.bitwise_and(bg, bg, mask=~person)
    img = cv2.add(tmp1,tmp2)

    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print("total runtime: "+str(t))

    cv2.imshow("input",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

#!/bin/bash
cd bodypix_mobilenet_float_050_model-stride16
echo "Converting TFJS to SavedModel..."
~/.local/bin/tfjs_graph_converter --output_format tf_saved_model . savedmodel
echo "Fixing SavedModel SignatureDefs..."
python3 ../myconverter.py
echo "Converting fixed SavedModel to TFLite..."
~/.local/bin/toco --saved_model_dir savedmodel_signaturedefs/ --output_file bodypix.tflite


#!/bin/bash
cd "$1" || exit
rm -r *.tflite ./savedmodel*
echo "Converting TFJS to SavedModel..."
~/.local/bin/tfjs_graph_converter --output_format tf_saved_model . savedmodel/
echo "Fixing SavedModel SignatureDefs..."
python3 ../myconverter.py
echo "Checking SignatureDefs..."
~/.local/bin/saved_model_cli show --dir savedmodel_signaturedefs/ --all
#echo "Converting fixed SavedModel to TFLite..."
#~/.local/bin/toco --saved_model_dir savedmodel_signaturedefs/ --output_file bodypix.tflite


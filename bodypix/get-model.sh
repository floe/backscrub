#! /bin/bash

# Exit on any error
set -e

function usage() {
  cat <<_EOT_
Usage:
  $0 model-name

Options:
  model-name  Specify a model name to download. (e.g. bodypix/resnet50/float/model-stride16)
               Refer to https://storage.googleapis.com/tfjs-models for the available models.

_EOT_
  exit 1
}

# Check args
[ -z $1 ] && usage

# Define constants & variables
BASE_URL=https://storage.googleapis.com/tfjs-models/savedmodel
MODEL_NAME=$1
DIR_NAME=$(echo ${MODEL_NAME} | tr "/" "_")
JQ=$(which jq || :)

# Verify jq is installed
[ -z ${JQ} ] && echo 'Please install "jq".' && exit 1

# Fetch model.json and weights.bin
mkdir ${DIR_NAME}

pushd ${DIR_NAME}
wget -c -nv ${BASE_URL}/${MODEL_NAME}.json -O model.json
cat model.json |
  ${JQ} -r ".weightsManifest | map(.paths) | flatten | @csv" |
  tr "," "\n" |
  xargs -I% wget -c ${BASE_URL}/${MODEL_NAME%/*}/%
popd

echo "Successfully downloaded to: ${DIR_NAME}"

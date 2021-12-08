#!/usr/bin/env bash

(

if [[ " $* " =~ " "("-h"|"--help")" " ]]
then
  echo "Update and (re-)build backscrub"
  echo "USAGE:"
  echo "  ./build.sh [OPTIONS]"
  echo ""
  echo "  Options:"
  echo "    --h, --help  Show this help message and exit"
  echo "    --no-update  Skip fetching updates from github"
  exit 0
fi

# Get the root directory of the repository
wd="$(cd "$(dirname "$0")" || exit 1; pwd)"

echo "working directory: $wd"
cd "$wd"

# Make sure we have the latest version of backscrub
echo "Fetching latest version..."
[[ " $* " =~ " --no-update" ]] || git pull --recurse-submodules
echo "..done"
echo ""


# Backup the current build directory (if present)
if [[ -d "./build" ]]
then
  echo "Backing up build directory..."
  [[ ! -d "./build_bk" ]] || rm -rf "./build_bk"
  mv ./build ./build_bk
  echo "..done"
  echo ""
fi

# Restore the backup on error
trap 'rm -rf ./build && mv ./build_bk ./build' SIGHUP SIGINT SIGTERM

# Abort on errors
set -e

echo "Building backscrub..."

#mkdir build
cd build

# Build backscrub
cmake ..
make -j "$(nproc || echo 4)"
ln -s ../models models

echo "..done"
echo ""

echo "Cleaning up..."
# Delete the backup on success
[[ ! -d "./build_bk" ]] || rm -rf "./build_bk"
echo "..done"
echo ""

echo "All done. You can run backscrub like this:"
echo "    cd \"$wd/build\" && ./backscrub -h"
echo ""
echo "Or you can add the following line to your ~/.bashrc, to get a backscrub command:"
echo "backscrub() { pushd \"$wd/build\" > /dev/null && { ./backscrub \"\$@\"; popd > /dev/null; }; }"


)


#!/usr/bin/env bash

# Abort on errors
set -e

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
backup_dir="$(mktemp -d)"

if [[ -d "./build" ]]
then
  echo "Backing up build directory..."
  mv ./build "$backup_dir/build"
  echo "..done"
  echo ""
  echo "backup dir: $backup_dir"

  restore_backup() {
    echo 'Restoring backup...'
    if [[ -d "${1?}/build" ]] 
    then
      rm -rf ./build && mv "$1/build" ./build
    fi
    rm -rf "$1"
    echo '..done'
  }

  # Restore the backup on error
  trap "restore_backup $backup_dir" ERR

fi

echo "Building backscrub..."

mkdir -p ./build
cd ./build

# Build backscrub
cmake ..
make -j "$(nproc || echo 4)"
ln -s ../models models

echo "..done"
echo ""

if [[ -d "$backup_dir" ]];
then
  echo "Cleaning up..."
  # Delete the backup on success
  rm -rf "$backup_dir"
  echo "..done"
  echo ""
fi

echo "All done. You can run backscrub like this:"
echo "    cd \"$wd/build\" && ./backscrub -h"
echo ""
echo "Or you can add the following line to your ~/.bashrc, to get a backscrub command:"
echo "backscrub() { ( cd \"$wd/build\" && ./backscrub \"\$@\" ) }"



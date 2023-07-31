#!/bin/bash

# Get the path of the current script
script_path="$( cd "$(dirname "$0")" ; pwd -P )"

# Set the target location to be two directories above the current location
target_location="$script_path/python/tvm/libtvm.so"

url="https://dl.espressif.com/AI/esp_dl/libtvm.so"

wget -O $target_location $url

if [ $? -eq 0 ]; then
    echo "File successfully downloaded."
else
    echo "Failed to download the file."
fi

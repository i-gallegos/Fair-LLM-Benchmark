#/bin/bash

set -e

echo "======================================="
echo    "      DOWNLOADING DATA"
echo "======================================="


wget https://ai2-public-datasets.s3.amazonaws.com/unqover/data.zip

unzip data.zip

echo "Removing data.zip"
rm data.zip

echo "Downloading Complete!"

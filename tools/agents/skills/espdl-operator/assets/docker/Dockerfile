FROM espressif/idf:v5.4.3

# Install build dependencies
RUN apt-get update && apt-get install -y ninja-build

ENV IDF_PATH=/opt/esp/idf
ENV IDF_PATH_FORCE=1

# Copy and install Python dependencies (excluding esp-ppq, which will be mounted from source)
COPY docker_requirements.txt /tmp/requirements.txt
RUN . $IDF_PATH/export.sh && pip install --no-cache-dir -r /tmp/requirements.txt

# esp-ppq will be mounted at /esp-ppq and installed in editable mode at runtime
# This allows local modifications to take effect immediately without rebuilding the image

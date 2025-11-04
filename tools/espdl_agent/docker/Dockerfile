FROM espressif/idf:v5.4.3

#install PyTorch and  esp-ppq
RUN apt-get update && apt-get install -y ninja-build
ENV IDF_PATH=/opt/esp/idf
ENV IDF_PATH_FORCE=1
COPY docker_requirements.txt /tmp/requirements.txt
RUN . $IDF_PATH/export.sh && pip install --no-cache-dir -r /tmp/requirements.txt

#/bin/bash

# The working directory
export HOST_WORK_DIR="$(pwd)"

# get host user id to populate to the container
export HOST_USER_ID="$(id -u)"
export HOST_GROUP_ID="$(id -g)"
export HOST_USER_NAME=${USER}
export CONTAINER_WORKDIR="/workspace/rhsp_lsnr/"

docker run -it --runtime=nvidia --rm --name rhsp_lsnr \
-v "$HOST_WORK_DIR":"$CONTAINER_WORKDIR" \
-e CONTAINER_UID=${HOST_USER_ID} \
-e CONTAINER_GID=${HOST_GROUP_ID} \
-e CONTAINER_UNAME=${HOST_USER_NAME} \
-e CONTAINER_WORKDIR=${CONTAINER_WORKDIR} \
rhsp_lsnr:1.0.0
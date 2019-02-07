#! /bin/bash
if [ $# -gt 0 ]
then    CONTAINER_NAME="--name $1";  echo "After you exit the container you can resume it issuing \" docker start $1 \"";
else    CONTAINER_NAME="";
fi
DEVICE="gpu"
nvidia-docker run -v $PWD:/home/DeepMRI $CONTAINER_NAME -it edoardogiacomello/deepmri:latest-$DEVICE



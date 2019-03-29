#! /bin/bash
if [ $# -gt 0 ]
then    CONTAINER_NAME="--name $1";  echo "After you exit the container you can resume it issuing \" docker start $1 \"";
else    CONTAINER_NAME="";
fi
echo "Mapping container port 8888 to host port 8889"
DEVICE="gpu"
nvidia-docker run -p 8889:8888 -v $PWD/../datasets/:/home/datasets/:ro -v $PWD:/home/DeepMRI $CONTAINER_NAME -it edoardogiacomello/deepmri:latest-$DEVICE



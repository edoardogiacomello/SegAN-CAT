#! /bin/bash
if [ $# -gt 0 ] 
then	CONTAINER_NAME="--name $1"; echo "After you exit the container you can resume it issuing \" docker start $1 \"";
else	CONTAINER_NAME="";
fi
DEVICE="cpu"
docker run -v $PWD:/home/DeepMRI -v $PWD/../datasets/:/home/datasets/:ro -p 8888:8888 -p 6006:6006 $CONTAINER_NAME -it edoardogiacomello/deepmri:latest-$DEVICE



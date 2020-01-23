#! /bin/bash
if [ $# -gt 0 ]
then    CONTAINER_NAME="--name $1";  echo "After you exit the container you can resume it issuing \" docker start $1 \"";
else    CONTAINER_NAME="";
fi
echo "Mapping container port 8888 to host port 8889"
nvidia-docker run -p 8887:8888 -p 6006:6006 -v $PWD/../datasets/:/tf/datasets/:ro -v $PWD:/tf/DeepMRI --name=deepmri2 -it edoardogiacomello/deepmri:latest-gpu


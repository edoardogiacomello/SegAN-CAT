ARG TF_TAG=latest-gpu-py3
FROM tensorflow/tensorflow:$TF_TAG
ADD . /home/DeepMRI/
WORKDIR /home/DeepMRI/
RUN apt-get update
RUN apt-get install -y vim
# Installing required packages (some are already present in tensorflow distribution)
RUN pip install --upgrade request scikit-image scikit-learn seaborn tensorflow-probability SimpleITK
# Exposing port 6006 to host for tensorboard
EXPOSE 6006
CMD bash -C 'dockerrc.sh'; '/bin/bash'

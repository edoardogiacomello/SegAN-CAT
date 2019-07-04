ARG TF_TAG=1.12.0-gpu-py3
FROM tensorflow/tensorflow:$TF_TAG
ADD . /home/DeepMRI/
WORKDIR /home/DeepMRI/
RUN apt-get update
RUN apt-get install -y vim wget
# Installing nodejs required for jupyterlab plugins
RUN wget https://deb.nodesource.com/setup_11.x
RUN chmod +x setup_11.x
RUN bash setup_11.x
RUN apt-get install -y nodejs
# Installing required packages (some are already present in tensorflow distribution)
RUN pip install --upgrade request scikit-image scikit-learn seaborn tensorflow-probability SimpleITK jupyterlab ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets
# Exposing port 6006 to host for tensorboard
EXPOSE 6006
CMD bash -C 'dockerrc.sh'; '/bin/bash'

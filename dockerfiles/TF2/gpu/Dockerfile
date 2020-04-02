ARG TF_TAG=2.0.0b1-gpu-py3
FROM tensorflow/tensorflow:$TF_TAG
WORKDIR /tf
RUN apt-get update
RUN apt-get install -y vim wget tmux
# Installing nodejs required for jupyterlab plugins
RUN wget https://deb.nodesource.com/setup_11.x
RUN chmod +x setup_11.x
RUN bash setup_11.x
RUN apt-get install -y nodejs
# Installing required packages (some are already present in tensorflow distribution)
RUN pip install --upgrade request scikit-image scikit-learn pandas seaborn tensorflow-probability SimpleITK jupyterlab ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
# Installing tensorboard for jupyterlab
RUN pip install --upgrade jupyter-tensorboard
RUN jupyter labextension install jupyterlab_tensorboard
# This is to fix a bug in tensorflow docker not finding nvidia drivers:
RUN ldconfig
CMD bash

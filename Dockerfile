# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

# scipy/machine learning
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.1-stable

# from base notebook
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:latest

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# change to root to install packages
USER root

# CUDA Toolkit
RUN conda install -y cudatoolkit=10.1 cudnn nccl && \
    conda clean --all -f -y

# install Pytorch 1.7.*
# Copy-paste command from https://pytorch.org/get-started/locally/#start-locally
# Use the options stable, linux, pip, python and appropriate CUDA version
RUN pip install --no-cache-dir \
    torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# install packages
RUN pip install --no-cache-dir dgl-cu101 "torchtext<0.9.0"

# 4) change back to notebook user
# COPY /run_jupyter.sh /
# RUN chmod 755 /run_jupyter.sh
# USER $NB_UID

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]
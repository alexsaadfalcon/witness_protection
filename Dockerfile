# SageMaker PyTorch Image
# using 'us-east-1' region because that is my (Derin) aws account region
FROM 520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-pytorch:0.4.0-gpu-py3

COPY simple-faster-rcnn-pytorch/requirements.txt /tmp/

# install requirements for simple faster rcnn pytorch
RUN pip install -r /tmp/requirements.txt
RUN pip list

ENV PATH="/opt/ml/code:${PATH}"

COPY simple-faster-rcnn-pytorch /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV SAGEMAKER_PROGRAM train.py

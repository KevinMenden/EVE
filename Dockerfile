FROM nvidia/cuda

COPY * opt/eve

RUN cd opt/eve; pip install -e .



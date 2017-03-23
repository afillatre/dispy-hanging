FROM ubuntu:16.10

MAINTAINER Alexandre FILLATRE <afillatre@ippon.fr>

RUN apt-get update && apt-get install -y \
    python3-pip

RUN pip3 install --upgrade pip && \
    pip3 install psutil pandas sklearn scipy && \
    python3.5 -m pip install dispy

WORKDIR /jobs

ENTRYPOINT ["python3.5", "/usr/local/bin/dispynode.py"]

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
  python3-dev \
  python3 \
  python3-pip \
  gcc \
  build-essential \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
  && python3 -m pip install --no-cache-dir -r /requirements.txt \
  && rm /requirements.txt

COPY . /workspaces/app

CMD ["/usr/bin/python3", "/workspaces/app/app/main.py"]
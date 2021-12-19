FROM nvidia/cuda:11.4.2-base-ubuntu20.04 as base
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential python3 pip && \ 
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip
WORKDIR /home/tests/
COPY requirements.txt python/* ./
RUN pip install --no-cache-dir -r requirements.txt

FROM base as test
RUN make automate_train_probes

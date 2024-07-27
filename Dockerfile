ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# GNU compiler
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        gfortran && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt \
    setup.cfg \
    versioneer.py \
    .gitattributes \
    setup.py \
    MANIFEST.in \
    README.md \
    /opt/code/PyMAIA/

COPY PyMAIA/ \
    /opt/code/PyMAIA/PyMAIA/

COPY PyMAIA_scripts/* \
    /opt/code/PyMAIA/PyMAIA_scripts/

WORKDIR /opt/code/PyMAIA

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         && \
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install /opt/code/PyMAIA

ENTRYPOINT []



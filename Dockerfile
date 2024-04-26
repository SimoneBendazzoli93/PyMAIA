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
    main.py \
    inference.py \
    MLproject \
    MANIFEST.in \
    README.md \
    /opt/code/PyMAIA/

COPY PyMAIA/ \
    /opt/code/PyMAIA/PyMAIA/

COPY PyMAIA_scripts/* \
    /opt/code/PyMAIA/PyMAIA_scripts/

COPY ./bundles/ \
    /opt/code/PyMAIA/bundles/

WORKDIR /opt/code/PyMAIA

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         && \
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install -e /opt/code/PyMAIA

ENTRYPOINT []



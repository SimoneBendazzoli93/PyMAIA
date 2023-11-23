FROM ${BASE_IMAGE}

# GNU compiler
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        gfortran && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt \
    setup.cfg \
    versioneer.py \
    .gitattributes \
    setup.py \
    MANIFEST.in \
    README.md \
    /opt/code/Hive/

COPY ./docs/source/apidocs/configs/nnUNet_config_template.json \
    /opt/code/Hive/docs/source/apidocs/configs/nnUNet_config_template.json

COPY ./Hive/ \
    /opt/code/Hive/Hive/

COPY ./scripts/* \
    /opt/code/Hive/scripts/

WORKDIR /opt/code/Hive

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         && \
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install -e /opt/code/Hive

ENTRYPOINT []



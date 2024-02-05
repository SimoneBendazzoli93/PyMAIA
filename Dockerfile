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
    main.py \
    MLproject \
    MANIFEST.in \
    README.md \
    /opt/code/Hive/

COPY ./Hive/ \
    /opt/code/Hive/Hive/

COPY ./scripts/* \
    /opt/code/Hive/scripts/

COPY ./bundles/ \
    /opt/code/Hive/bundles/

WORKDIR /opt/code/Hive

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         && \
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install -e /opt/code/Hive

ENTRYPOINT []



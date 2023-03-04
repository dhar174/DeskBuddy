# -----------------------------------------------------------------------------
# Dockerfile for Debian package of Larynx (https://github.com/rhasspy/larynx)
#
# Requires Docker buildx: https://docs.docker.com/buildx/working-with-buildx/
# See scripts/build-docker.sh
# -----------------------------------------------------------------------------
FROM debian:bullseye as build
ARG TARGETARCH
ARG TARGETVARIANT

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
    build-essential libssl1.1 dpkg-dev python3 python3-pip python3-venv python3-dev

ENV APP_DIR=/usr/lib/larynx-tts
RUN python3 -m venv ${APP_DIR}
ENV PYTHON ${APP_DIR}/bin/python3
RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    ${PYTHON} -m pip install --upgrade pip && \
    ${PYTHON} -m pip install --upgrade wheel setuptools

COPY download/ /download/
COPY requirements.txt ${APP_DIR}/

RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    cat ${APP_DIR}/requirements.txt | \
    grep -v '^torch' | \
    xargs ${PYTHON} -m pip install \
    -f /download/ \
    -f 'https://synesthesiam.github.io/prebuilt-apps/'

# Create Debian package
ENV BUILD_DIR=/build
ENV BUILD_APP_DIR=${BUILD_DIR}/larynx-tts/${APP_DIR}

RUN mkdir -p ${BUILD_DIR}/larynx-tts/usr/lib && \
    cp -R ${APP_DIR} ${BUILD_DIR}/larynx-tts/usr/lib/

# Copy default voices/vocoders
COPY local/ ${BUILD_DIR}/larynx-tts/${APP_DIR}/local/

# Copy Larynx source
COPY licenses/ ${BUILD_APP_DIR}/licenses/
COPY larynx/ ${BUILD_APP_DIR}/larynx/
COPY README.md LICENSE ${BUILD_APP_DIR}/

COPY debian/control.in /

ENV DEBIAN_ARCH=${TARGETARCH}${TARGETVARIANT}
RUN export VERSION="$(cat ${BUILD_DIR}/larynx-tts/${APP_DIR}/larynx/VERSION)" && \
    if [ "${DEBIAN_ARCH}" = 'armv7' ]; then \
    export DEBIAN_ARCH='armhf'&& \
    sed -i 's/^Depends: /Depends: libatlas3-base,libgfortran5,/' /control.in ; \
    fi && \
    mkdir -p ${BUILD_DIR}/larynx-tts/DEBIAN && \
    sed -e s"/@VERSION@/${VERSION}/" -e "s/@DEBIAN_ARCH@/${DEBIAN_ARCH}/" < /control.in > ${BUILD_DIR}/larynx-tts/DEBIAN/control

COPY debian/larynx debian/larynx-server ${BUILD_DIR}/larynx-tts/usr/bin/
RUN chmod +x ${BUILD_DIR}/larynx-tts/usr/bin/*

RUN cd ${BUILD_DIR} && \
    dpkg --build larynx-tts

RUN cd ${BUILD_DIR} && \
    dpkg-name *.deb

# -----------------------------------------------------------------------------

FROM ubuntu

COPY --from=build /build/*.deb /

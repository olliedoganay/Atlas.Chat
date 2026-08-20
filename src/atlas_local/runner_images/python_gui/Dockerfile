FROM docker.io/library/python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG ATLAS_RUNTIME_DEFINITION_SHA256

LABEL org.opencontainers.image.title="Atlas Python GUI Runtime" \
      org.opencontainers.image.version="1.0.1" \
      com.atlas.runner.runtime="python-gui" \
      com.atlas.runner.runtime.version="1.0.1" \
      com.atlas.runner.runtime.definition-sha256="${ATLAS_RUNTIME_DEFINITION_SHA256}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        fluxbox \
        fontconfig \
        fonts-dejavu-core \
        libasound2 \
        libfreetype6 \
        libportmidi0 \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 \
        ncurses-term \
        novnc \
        tcl8.6 \
        tk8.6 \
        util-linux \
        websockify \
        x11vnc \
        xterm \
        xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html \
    && find -P /usr/share/novnc -type d -exec chmod 0555 {} + \
    && find -P /usr/share/novnc -type f -exec chmod 0444 {} +

COPY runtime-requirements.txt /opt/atlas/runtime-requirements.txt

RUN python -m pip install \
        --no-cache-dir \
        --no-input \
        --only-binary=:all: \
        --require-hashes \
        --requirement /opt/atlas/runtime-requirements.txt \
    && python -c "import numpy, pygame; assert numpy.__version__ == '2.5.2'; assert pygame.version.ver == '2.6.1'" \
    && chmod -R a-w /opt/atlas

USER 65534:65534
WORKDIR /work

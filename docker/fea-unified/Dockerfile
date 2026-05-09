FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install common build/runtime toolchains used across FEA-Bench repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    jq \
    unzip \
    zip \
    tar \
    xz-utils \
    build-essential \
    cmake \
    pkg-config \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    openjdk-17-jdk \
    maven \
    ripgrep \
    fd-find \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (for JS/TS build and test tasks).
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest pnpm yarn \
    && rm -rf /var/lib/apt/lists/*

# Keep Python packaging tools up-to-date.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /testbed

CMD ["bash"]

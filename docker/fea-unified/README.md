# Unified FEA-Bench Runtime Image

This image is a single baseline runtime for running mixed FEA-Bench projects in Docker with mini-swe-agent.

Included toolchains:
- Python 3 + pip/venv/dev headers
- Node.js 20 + npm/pnpm/yarn
- OpenJDK 17 + Maven
- Common build tools (gcc/g++, make, cmake, pkg-config)
- Git/Git LFS and common CLI tools

## Build Locally

```bash
cd benchmarks/mini-swe-agent/docker/fea-unified
bash build.sh fea-unified:latest
```

## Smoke Test

```bash
docker run --rm -it fea-unified:latest bash -lc 'python3 --version && node --version && java -version && mvn -version'
```

## Use With mini-swe-agent

Use this image as a fixed `docker_image` value in every dataset instance, or set it in config as a fallback image.

Example config fragment:

```yaml
environment:
  environment_class: docker
  image: fea-unified:latest

run:
  env_startup_command: |
    set -e
    rm -rf /testbed/*
    git clone https://github.com/{{ repo }} /testbed
    cd /testbed
    git checkout {{ base_commit }}
    if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi
    if [ -f pyproject.toml ]; then pip3 install -e . || true; fi
```

Notes:
- If your dataset provides `docker_image` per instance, that value takes precedence in the default swebench runner.
- For private repos, forward credentials via environment variables in mini config.

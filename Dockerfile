# Base image for the envlib ingest family: the RELEASED toolkit + its full stack, straight
# from PyPI — the package's own pyproject is the single source of truth for dependencies
# (no separately-maintained requirements files to drift; the cfdb >= 0.9.4 encoder-fix
# floor is enforced by the toolkit's requires_dist). Downstream ingest repos build
# `FROM envlib-ingest-base:<tag>` and add only their source + their own extra deps.
FROM python:3.12-slim

# NO default on purpose: the version must always be passed explicitly (docker-compose.yml
# derives it from envlib_ingest_base/__init__.py — see the build command there). A missing
# or empty value fails loudly at the ${...:?} guard below instead of building a stale image.
ARG TOOLKIT_VERSION

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir "envlib-ingest-base==${TOOLKIT_VERSION:?pass the TOOLKIT_VERSION build-arg (see docker-compose.yml)}"

CMD ["python"]

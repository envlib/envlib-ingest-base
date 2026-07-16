# Base image for the envlib ingest family: the envlib stack + this shared toolkit, pre-installed.
# Downstream ingest repos build `FROM envlib-ingest-base:<tag>` and add only their source.
FROM python:3.12-slim

RUN pip install --no-cache-dir -U pip

# 1. slow-moving scientific base (own layer -> cached across stack bumps)
COPY requirements_base.txt ./
RUN pip install --no-cache-dir -r requirements_base.txt

# 2. the envlib stack (faster-moving)
COPY requirements_envlib.txt ./
RUN pip install --no-cache-dir -r requirements_envlib.txt

# 3. the toolkit itself (deps already satisfied above)
COPY pyproject.toml README.md ./
COPY envlib_ingest_base ./envlib_ingest_base
RUN pip install --no-cache-dir --no-deps .

CMD ["python"]

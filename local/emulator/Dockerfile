FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS build

WORKDIR /build
COPY local/emulator/app.py /build/app.py
COPY local/emulator/push_gateway.py /build/push_gateway.py
RUN python -m py_compile /build/app.py /build/push_gateway.py

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && install -d -o app -g app /app /data
COPY --from=build --chown=app:app /build/app.py /app/app.py
COPY --from=build --chown=app:app /build/push_gateway.py /app/push_gateway.py
USER 10001:10001
WORKDIR /app
EXPOSE 8090
CMD ["python", "/app/app.py"]

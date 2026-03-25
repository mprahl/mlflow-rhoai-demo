FROM registry.access.redhat.com/ubi10/nodejs-22

USER 0

WORKDIR /opt/app-root/src

RUN microdnf install -y python3.12 python3.12-pip git \
    && microdnf clean all \
    && python3.12 -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md langgraph.json ./
COPY mlflow_notes_agent ./mlflow_notes_agent

RUN uv sync --frozen --group dev

COPY ui/package.json ui/package-lock.json ./ui/
RUN npm ci --prefix ui

COPY ui ./ui
RUN npm run build --prefix ui

ENV PATH="/opt/app-root/src/.venv/bin:${PATH}"

RUN chown -R 1001:0 /opt/app-root/src

USER 1001

EXPOSE 2024 3000

CMD ["uv", "run", "mlflow-notes-agent-serve", "--host", "0.0.0.0", "--port", "2024"]

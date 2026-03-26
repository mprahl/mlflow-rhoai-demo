FROM registry.access.redhat.com/ubi10/nodejs-22 AS python-builder

USER 0

WORKDIR /opt/app-root/src

RUN dnf install -y python3.12 python3.12-pip git \
    && dnf clean all \
    && python3.12 -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md langgraph.json ./
COPY mlflow_notes_agent ./mlflow_notes_agent

RUN uv sync --frozen --no-dev --no-editable

FROM registry.access.redhat.com/ubi10/nodejs-22 AS ui-builder

USER 0

WORKDIR /opt/app-root/src/ui

COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui ./
RUN npm run build

FROM registry.access.redhat.com/ubi10/nodejs-22

USER 0

WORKDIR /opt/app-root/src

RUN dnf install -y python3.12 \
    && dnf clean all

COPY --from=python-builder --chown=1001:0 /opt/app-root/src/.venv ./.venv
COPY --chown=1001:0 pyproject.toml README.md langgraph.json ./
COPY --chown=1001:0 mlflow_notes_agent ./mlflow_notes_agent
COPY --from=ui-builder --chown=1001:0 /opt/app-root/src/ui/.next/standalone ./ui
COPY --from=ui-builder --chown=1001:0 /opt/app-root/src/ui/.next/static ./ui/.next/static

ENV PATH="/opt/app-root/src/.venv/bin:${PATH}"

USER 1001

EXPOSE 2024 3000

CMD ["/opt/app-root/src/.venv/bin/mlflow-notes-agent-serve", "--host", "0.0.0.0", "--port", "2024"]

# MLflow Agent Notes Demo

A minimal demo for showing MLflow agent tracing on Red Hat OpenShift AI with:

- a Python LangGraph backend
- Gemini via LangChain
- a bare-bones `assistant-ui` chat frontend
- GitHub-backed markdown notes
- MLflow traces logged to the `mlflow-demo` experiment

## What It Does

The assistant can:

- search the web with DuckDuckGo
- search Wikipedia
- answer with awareness of today's date
- list saved notes from GitHub
- read saved notes from GitHub
- delete saved notes from GitHub
- save a new markdown note into the configured GitHub repo and branch

Each saved note is created with a filename like `YYYY-MM-DD-summary.md`.

## Architecture

- `mlflow_notes_agent/`: LangGraph agent, tools, GitHub note storage, and MLflow tracing startup
- `langgraph.json`: LangGraph graph registration and FastAPI app hook
- `ui/`: Next.js + `assistant-ui` chat client connected to the LangGraph API with `@langchain/langgraph-sdk`

## Quick Start

### 1. Create your env files

Copy the examples and fill in your values:

```bash
cp .env.example .env
```

Required backend values:

- `GOOGLE_API_KEY`
- `GITHUB_TOKEN`
- `GITHUB_REPO`
- `GITHUB_BRANCH`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_AUTH`
- `MLFLOW_WORKSPACE`

The `GITHUB_BRANCH` branch should already exist in the target repository.
For this demo, an empty `notes` branch works well.

Optional frontend overrides:

- `NEXT_PUBLIC_LANGGRAPH_API_URL`
- `NEXT_PUBLIC_LANGGRAPH_ASSISTANT_ID`

For local development, the frontend defaults to:

- `NEXT_PUBLIC_LANGGRAPH_API_URL=http://127.0.0.1:2024`
- `NEXT_PUBLIC_LANGGRAPH_ASSISTANT_ID=agent`

If you want to override those values, copy the example file:

```bash
cp ui/.env.local.example ui/.env.local
```

### 2. Install dependencies

The backend intentionally installs MLflow from a git commit because the demo
needs the auth plugin included in that revision.

```bash
uv sync --group dev
cd ui && npm install
```

### 3. Start the backend

```bash
uv run mlflow-notes-agent-serve
```

That starts the LangGraph API on `http://127.0.0.1:2024` by default.

### 4. Start the frontend

In a second terminal:

```bash
cd ui
npm run dev
```

Then open [http://127.0.0.1:3000](http://127.0.0.1:3000).

## Demo Flow

1. Ask the assistant to research a topic.
2. Let it use DuckDuckGo and Wikipedia tools.
3. Ask it to save the result as a note.
4. Open MLflow and inspect the trace for the LLM call and tool spans.
5. Verify the markdown file was created in the configured GitHub repo and branch.

## Example Prompts

- `Research MLflow agent tracing on OpenShift AI and save a short note.`
- `List the notes already saved in GitHub.`
- `Read the latest note and summarize it.`
- `Create a note about Gemini tracing in MLflow with a few source links.`

## OpenShift Notes

This repo is set up so the same code can run locally or in OpenShift by
injecting environment variables into the backend and frontend containers.

- The `manifests/` directory deploys one Pod with two containers: the LangGraph
  backend and the standalone Next.js UI.
- The UI container proxies `/api/langgraph/*` to `http://127.0.0.1:2024`, so
  only the UI needs an OpenShift Route.
- The root `Containerfile` uses multi-stage builds so the final UBI 10-based
  runtime image keeps only the Python runtime environment and the built
  standalone Next.js frontend.
- Override the container command in Kubernetes manifests depending on whether
  the workload should run the LangGraph backend or the Next.js frontend.
- The default container command starts the LangGraph backend. To run the
  frontend from the same image, override the command to `node ui/server.js`.
- The backend only depends on env vars for GitHub, Gemini, and MLflow settings.
- The frontend only needs the public LangGraph API URL and assistant id.
- MLflow tracing is configured at startup with `mlflow.langchain.autolog()`.
- `mlflow-notes-agent-judge --trace-id <trace_id>` runs two Gemini-powered MLflow judges
  against a trace: `tool_accuracy_ok` and `low_user_frustration`.
- `mlflow-notes-agent-judge --experiment-name mlflow-demo --max-traces 20`
  fetches traces from that experiment and scores only the ones still missing
  one or both judge assessments.

### Deploy To OpenShift

Create the namespace once:

```bash
oc new-project mlflow-demo-agent
```

Create or replace the backend Secret directly from `.env`:

```bash
oc project mlflow-demo-agent
oc create secret generic mlflow-notes-agent-env \
  --from-env-file=.env \
  --dry-run=client -o yaml | oc apply -f -
```

The OpenShift deployment overrides `MLFLOW_TRACKING_URI` to the in-cluster
`mlflow.redhat-ods-applications.svc` service and sets `REQUESTS_CA_BUNDLE` to
the projected OpenShift service CA. That lets the Secret still come from your
existing `.env` while avoiding the external OAuth-protected gateway route from
inside the cluster. This is to avoid a bug in RHOAI 3.3 where service account tokens don't work on the gateway.

The manifests also create a dedicated `mlflow-notes-agent` ServiceAccount in
`mlflow-demo-agent` and bind it to a Role in the `mlflow-demo` workspace
namespace so the pod can create and update MLflow experiments and traces using
the Kubernetes-backed MLflow authorization plugin.

Apply the manifests:

```bash
oc apply -k manifests
```

Watch the rollout:

```bash
oc rollout status deploy/mlflow-notes-agent
```

Get the UI Route:

```bash
oc get route mlflow-notes-agent-ui -o jsonpath='https://{.spec.host}{"\n"}'
```

## Important Files

- `pyproject.toml`
- `Containerfile`
- `langgraph.json`
- `mlflow_notes_agent/agent.py`
- `mlflow_notes_agent/judges.py`
- `mlflow_notes_agent/tools.py`
- `mlflow_notes_agent/github_notes.py`
- `mlflow_notes_agent/webapp.py`
- `ui/app/assistant.tsx`
- `ui/lib/chatApi.ts`

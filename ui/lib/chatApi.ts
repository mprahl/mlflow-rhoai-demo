import { Client, ThreadState } from "@langchain/langgraph-sdk";
import {
  LangChainMessage,
  LangGraphCommand,
} from "@assistant-ui/react-langgraph";

const DEFAULT_LANGGRAPH_API_URL = "/api/langgraph";
const DEFAULT_ASSISTANT_ID = "agent";

const getApiUrl = () => {
  return (
    process.env["NEXT_PUBLIC_LANGGRAPH_API_URL"]?.trim() ||
    DEFAULT_LANGGRAPH_API_URL
  );
};

const getAssistantId = () => {
  return (
    process.env["NEXT_PUBLIC_LANGGRAPH_ASSISTANT_ID"]?.trim() ||
    DEFAULT_ASSISTANT_ID
  );
};

const createClient = () => {
  return new Client({ apiUrl: getApiUrl() });
};

export const createThread = async () => {
  try {
    const client = createClient();
    return client.threads.create();
  } catch (error) {
    throw new Error(
      `Failed to connect to the LangGraph API at ${getApiUrl()}. Start the backend with 'uv run mlflow-notes-agent-serve' or set NEXT_PUBLIC_LANGGRAPH_API_URL.`,
      { cause: error },
    );
  }
};

export const getThreadState = async (
  threadId: string,
): Promise<ThreadState<{ messages: LangChainMessage[] }>> => {
  const client = createClient();
  return client.threads.getState(threadId);
};

export const sendMessage = async (params: {
  threadId: string;
  messages?: LangChainMessage[];
  command?: LangGraphCommand | undefined;
}) => {
  const client = createClient();
  return client.runs.stream(
    params.threadId,
    getAssistantId(),
    {
      input: params.messages?.length
        ? {
            messages: params.messages,
          }
        : null,
      command: params.command,
      streamMode: ["messages", "updates"],
    },
  );
};

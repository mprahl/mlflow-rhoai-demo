import { NextRequest } from "next/server";

const DEFAULT_LANGGRAPH_API_URL = "http://127.0.0.1:2024";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

const getLangGraphApiUrl = () => {
  return (
    process.env["LANGGRAPH_API_URL"]?.trim() ||
    process.env["NEXT_PUBLIC_LANGGRAPH_API_URL"]?.trim() ||
    DEFAULT_LANGGRAPH_API_URL
  ).replace(/\/$/, "");
};

const buildUpstreamUrl = (request: NextRequest, path: string[]) => {
  const upstreamUrl = new URL(`${getLangGraphApiUrl()}/${path.join("/")}`);
  upstreamUrl.search = new URL(request.url).search;
  return upstreamUrl;
};

const proxyRequest = async (request: NextRequest, context: RouteContext) => {
  const { path = [] } = await context.params;
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("content-length");

  const response = await fetch(buildUpstreamUrl(request, path), {
    method: request.method,
    headers,
    body: request.body,
    cache: "no-store",
    redirect: "manual",
    // Required by Node's fetch when forwarding streaming request bodies.
    duplex: "half",
  } as RequestInit & { duplex: "half" });

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
};

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const PATCH = proxyRequest;
export const DELETE = proxyRequest;
export const OPTIONS = proxyRequest;

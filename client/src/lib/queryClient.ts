import { QueryClient, QueryFunction } from "@tanstack/react-query";
import { getAuthHeaders } from "@/lib/auth";

const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? "";

function buildApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  if (BACKEND_URL.endsWith("/") && path.startsWith("/")) {
    return `${BACKEND_URL}${path.slice(1)}`;
  }
  return `${BACKEND_URL}${path}`;
}

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const authHeaders = getAuthHeaders() as Record<string, string>;
  const defaultHeaders: Record<string, string> = data ? { "Content-Type": "application/json" } : {};
  const headers: Record<string, string> = { ...defaultHeaders, ...authHeaders };

  const fullUrl = buildApiUrl(url);
  const res = await fetch(fullUrl, {
    method,
    headers,
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const authHeaders = getAuthHeaders();
    const url = buildApiUrl(queryKey.join("/"));
    const res = await fetch(url, {
      credentials: "include",
      headers: authHeaders,
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});

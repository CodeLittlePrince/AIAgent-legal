import type { Citation, Intent, StreamHandlers } from "../types";

const SESSION_KEY = "legal_assistant_session_id";

export function getStoredSessionId(): string | null {
  return localStorage.getItem(SESSION_KEY);
}

export function storeSessionId(sessionId: string): void {
  localStorage.setItem(SESSION_KEY, sessionId);
}

export function clearStoredSessionId(): void {
  localStorage.removeItem(SESSION_KEY);
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return { event, data: dataLines.join("\n") };
}

function dispatchEvent(event: string, payload: Record<string, unknown>, handlers: StreamHandlers) {
  switch (event) {
    case "session":
      handlers.onSession(String(payload.session_id), String(payload.trace_id));
      break;
    case "intent":
      handlers.onIntent(payload.intent as Intent);
      break;
    case "status":
      handlers.onStatus(String(payload.message ?? ""));
      break;
    case "delta":
      handlers.onDelta(String(payload.content ?? ""));
      break;
    case "citations":
      handlers.onCitations((payload.citations as Citation[]) ?? []);
      break;
    case "disclaimer":
      handlers.onDisclaimer(String(payload.disclaimer ?? ""));
      break;
    case "done":
      handlers.onDone();
      break;
    case "error":
      handlers.onError(String(payload.message ?? "Unknown error"));
      break;
    default:
      break;
  }
}

export async function streamChat(
  message: string,
  sessionId: string | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Empty response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const parsed = parseSseBlock(block.trim());
      if (!parsed) {
        continue;
      }
      const payload = parsed.data ? (JSON.parse(parsed.data) as Record<string, unknown>) : {};
      dispatchEvent(parsed.event, payload, handlers);
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseBlock(buffer.trim());
    if (parsed) {
      const payload = parsed.data ? (JSON.parse(parsed.data) as Record<string, unknown>) : {};
      dispatchEvent(parsed.event, payload, handlers);
    }
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/v1/sessions/${sessionId}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete session (${response.status})`);
  }
}

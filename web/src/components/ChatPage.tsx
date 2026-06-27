import { useEffect, useRef, useState } from "react";
import {
  clearStoredSessionId,
  deleteSession,
  getStoredSessionId,
  storeSessionId,
  streamChat,
} from "../api/chat";
import type { ChatMessage, Citation, Intent } from "../types";
import { ChatInput } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";

const starterMessages: ChatMessage[] = [
  {
    id: "welcome",
    role: "assistant",
    content:
      "你好，我是法律智能助手。你可以咨询劳动法、合同法等法律问题，查询城市天气，或进行日常对话。",
  },
];

function createId(): string {
  return crypto.randomUUID();
}

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [sessionId, setSessionId] = useState<string | null>(() => getStoredSessionId());
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text: string) => {
    if (isStreaming) {
      return;
    }

    setError(null);
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: text,
    };
    const assistantId = createId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        text,
        sessionId,
        {
          onSession: (nextSessionId: string) => {
            setSessionId(nextSessionId);
            storeSessionId(nextSessionId);
          },
          onIntent: (intent: Intent) => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId ? { ...item, intent, status: null } : item,
              ),
            );
          },
          onStatus: (statusMessage: string) => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId ? { ...item, status: statusMessage } : item,
              ),
            );
          },
          onDelta: (content: string) => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId
                  ? { ...item, content: item.content + content, status: null }
                  : item,
              ),
            );
          },
          onCitations: (citations: Citation[]) => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId ? { ...item, citations } : item,
              ),
            );
          },
          onDisclaimer: (disclaimer: string) => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId ? { ...item, disclaimer } : item,
              ),
            );
          },
          onDone: () => {
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId
                  ? { ...item, streaming: false, status: null }
                  : item,
              ),
            );
          },
          onError: (message: string) => {
            setError(message);
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      content: item.content || `出错了：${message}`,
                      streaming: false,
                    }
                  : item,
              ),
            );
          },
        },
        controller.signal,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "请求失败";
      setError(message);
      setMessages((prev) =>
        prev.map((item) =>
          item.id === assistantId
            ? { ...item, content: `出错了：${message}`, streaming: false }
            : item,
        ),
      );
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  };

  const handleNewChat = async () => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    if (sessionId) {
      try {
        await deleteSession(sessionId);
      } catch {
        // Best-effort cleanup; local state reset still proceeds.
      }
    }
    clearStoredSessionId();
    setSessionId(null);
    setMessages(starterMessages);
    setError(null);
    setIsStreaming(false);
  };

  return (
    <div className="chat-shell">
      <header className="chat-header">
        <div>
          <h1>法律智能助手</h1>
          <p>法律问答 · 天气查询 · 通用对话（SSE 流式）</p>
        </div>
        <button type="button" className="secondary" onClick={handleNewChat} disabled={isStreaming}>
          新对话
        </button>
      </header>

      <div className="chat-body" ref={listRef}>
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <footer className="chat-footer">
        <ChatInput disabled={isStreaming} onSend={handleSend} />
      </footer>
    </div>
  );
}

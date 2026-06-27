import type { ChatMessage, Citation, Intent } from "../types";
import { MarkdownContent } from "./MarkdownContent";

interface MessageBubbleProps {
  message: ChatMessage;
}

const intentLabels: Record<Intent, string> = {
  legal: "法律咨询",
  weather: "天气查询",
  general: "通用对话",
};

function formatSourceName(source: string): string {
  return source.replace(/\.md$/i, "").replace(/_/g, " ");
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const showStatus = !isUser && message.streaming && !message.content && message.status;
  const isLegalWithCitations =
    !isUser && message.intent === "legal" && message.citations && message.citations.length > 0;

  return (
    <article className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      {!isUser && message.intent && (
        <span className="intent-badge">{intentLabels[message.intent]}</span>
      )}

      {isLegalWithCitations && (
        <CitationPanel citations={message.citations!} />
      )}

      <div className="message-content">
        {showStatus ? (
          <p className="status-text">{message.status}</p>
        ) : isUser ? (
          message.content
        ) : (
          <MarkdownContent content={message.content} />
        )}
        {message.streaming && message.content && (
          <span className="cursor" aria-hidden="true" />
        )}
      </div>

      {!isUser && message.disclaimer && (
        <p className="disclaimer">{message.disclaimer}</p>
      )}
    </article>
  );
}

function CitationPanel({ citations }: { citations: Citation[] }) {
  const uniqueSources = [...new Set(citations.map((item) => item.source))];

  return (
    <section className="citation-panel" aria-label="法律依据引用">
      <div className="citation-panel-header">
        <span className="citation-icon" aria-hidden="true">
          📚
        </span>
        <div>
          <p className="citation-title">本次回答参考以下法律文档</p>
          <p className="citation-subtitle">
            共引用 {uniqueSources.length} 份文档，{citations.length} 条相关片段
          </p>
        </div>
      </div>
      <ul className="citation-source-list">
        {uniqueSources.map((source) => (
          <li key={source} className="citation-source-chip">
            {formatSourceName(source)}
          </li>
        ))}
      </ul>
      <div className="citation-excerpts">
        {citations.map((item, index) => (
          <article key={`${item.source}-${index}`} className="citation-card">
            <header className="citation-card-header">
              <span className="citation-index">{index + 1}</span>
              <strong>{formatSourceName(item.source)}</strong>
            </header>
            <p className="citation-excerpt">{item.excerpt}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

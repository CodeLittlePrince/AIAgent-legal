import ReactMarkdown from "react-markdown";

interface MarkdownContentProps {
  content: string;
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="md-p">{children}</p>,
          ul: ({ children }) => <ul className="md-ul">{children}</ul>,
          ol: ({ children }) => <ol className="md-ol">{children}</ol>,
          li: ({ children }) => <li className="md-li">{children}</li>,
          strong: ({ children }) => <strong className="md-strong">{children}</strong>,
          h1: ({ children }) => <h3 className="md-h">{children}</h3>,
          h2: ({ children }) => <h4 className="md-h">{children}</h4>,
          h3: ({ children }) => <h5 className="md-h">{children}</h5>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

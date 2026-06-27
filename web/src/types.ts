export type Intent = "legal" | "weather" | "general";

export interface Citation {
  source: string;
  excerpt: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: Intent;
  citations?: Citation[];
  disclaimer?: string | null;
  streaming?: boolean;
  status?: string | null;
}

export interface StreamHandlers {
  onSession: (sessionId: string, traceId: string) => void;
  onIntent: (intent: Intent) => void;
  onStatus: (message: string) => void;
  onDelta: (content: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDisclaimer: (disclaimer: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

import type { FormEvent, KeyboardEvent } from "react";

interface ChatInputProps {
  disabled: boolean;
  onSend: (message: string) => void;
}

export function ChatInput({ disabled, onSend }: ChatInputProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const input = form.elements.namedItem("message") as HTMLTextAreaElement;
    const value = input.value.trim();
    if (!value || disabled) {
      return;
    }
    onSend(value);
    input.value = "";
    input.style.height = "auto";
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const handleInput = (event: FormEvent<HTMLTextAreaElement>) => {
    const target = event.currentTarget;
    target.style.height = "auto";
    target.style.height = `${Math.min(target.scrollHeight, 160)}px`;
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <textarea
        name="message"
        placeholder="输入法律、天气或日常问题…（Enter 发送，Shift+Enter 换行）"
        rows={1}
        disabled={disabled}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
      />
      <button type="submit" disabled={disabled}>
        发送
      </button>
    </form>
  );
}

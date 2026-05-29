import { lazy, memo, Suspense } from "react";

type MessageContentProps = {
  content: string;
  streaming?: boolean;
};

const MarkdownMessageContent = lazy(async () => {
  const module = await import("./MarkdownMessageContent");
  return { default: module.MarkdownMessageContent };
});

export const MessageContent = memo(function MessageContent({ content, streaming = false }: MessageContentProps) {
  if (!looksLikeMarkdown(content)) {
    return <div className="message-content message-content-plain">{content}</div>;
  }

  return (
    <div className={`message-content${streaming ? " message-content-streaming" : ""}`}>
      <Suspense fallback={<div className="message-content-plain">{content}</div>}>
        <MarkdownMessageContent content={content} />
      </Suspense>
    </div>
  );
},
(previous, next) => previous.content === next.content && previous.streaming === next.streaming);

function looksLikeMarkdown(content: string) {
  return /(```|`|^\s*[-*]\s|^\s*\d+\.\s|^\s*#|\[[^\]]+\]\([^)]+\)|\|)/m.test(content);
}

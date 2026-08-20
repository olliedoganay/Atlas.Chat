import { ExternalLink } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { openExternalUrl } from "../lib/api";

type MarkdownMessageContentProps = {
  content: string;
};

const MarkdownCodeBlock = lazy(async () => {
  const module = await import("./MarkdownCodeBlock");
  return { default: module.MarkdownCodeBlock };
});

export function MarkdownMessageContent({ content }: MarkdownMessageContentProps) {
  const [linkError, setLinkError] = useState("");

  useEffect(() => {
    setLinkError("");
  }, [content]);

  const openMarkdownLink = async (url: string) => {
    setLinkError("");
    try {
      await openExternalUrl(url);
    } catch (error) {
      setLinkError(
        error instanceof Error ? error.message : "Atlas could not open this link safely.",
      );
    }
  };

  return (
    <>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const externalHref = safeExternalHref(href);
            if (!externalHref) {
              return <span className="markdown-link-disabled">{children}</span>;
            }
            return (
              <button
                className="markdown-external-link"
                onClick={() => void openMarkdownLink(externalHref)}
                role="link"
                title={`Open ${externalHref} in your browser`}
                type="button"
              >
                {children}
                <ExternalLink aria-hidden="true" size={12} />
              </button>
            );
          },
          code({ className, children, ...props }) {
            const raw = String(children).replace(/\n$/, "");
            const language = markdownCodeLanguage(className);
            if (!language) {
              return (
                <code className="inline-code" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <Suspense fallback={<StaticCodeBlock code={raw} />}>
                <MarkdownCodeBlock code={raw} language={language} />
              </Suspense>
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
      {linkError ? (
        <span className="error-inline markdown-link-error" role="alert">
          {linkError}
        </span>
      ) : null}
    </>
  );
}

export function markdownCodeLanguage(className?: string): string {
  return /(?:^|\s)language-([^\s]+)/.exec(className ?? "")?.[1] ?? "";
}

export function safeExternalHref(href?: string): string | null {
  const rawCandidate = href ?? "";
  const candidate = rawCandidate.trim();
  if (
    !candidate ||
    candidate !== rawCandidate ||
    candidate.length > 2048 ||
    /[\u0000-\u001f\u007f]/.test(candidate)
  ) {
    return null;
  }
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    if (parsed.username || parsed.password) {
      return null;
    }
    return parsed.toString();
  } catch {
    return null;
  }
}

function StaticCodeBlock({ code }: { code: string }) {
  return (
    <div className="code-block-shell">
      <div className="code-block-header">
        <span>code</span>
      </div>
      <div className="code-block-code" style={{ padding: "14px 16px", whiteSpace: "pre-wrap" }}>
        {code}
      </div>
    </div>
  );
}

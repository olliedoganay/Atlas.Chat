import { Check, Copy, Play } from "lucide-react";
import { useState } from "react";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/prism-light";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

import { openRunnerWindow, RUNNABLE_LANGUAGES, resolveRunnableLanguage } from "../lib/runner";
import { useAtlasStore } from "../store/useAtlasStore";

type MarkdownCodeBlockProps = {
  code: string;
  language: string;
};

SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("diff", diff);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("jsx", jsx);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);
SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("py", python);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("tsx", tsx);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);

export function MarkdownCodeBlock({ code, language }: MarkdownCodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [launching, setLaunching] = useState(false);
  const currentUserId = useAtlasStore((state) => state.currentUserId);
  const currentThreadId = useAtlasStore((state) => state.currentThreadId);
  const runnable = resolveRunnableLanguage(language);

  const copy = async () => {
    try {
      if (!navigator.clipboard?.writeText) {
        return;
      }
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard access can be denied in browser preview or locked-down desktop contexts.
    }
  };

  const run = async () => {
    if (!runnable || launching) {
      return;
    }
    setLaunching(true);
    try {
      await openRunnerWindow({
        language: runnable,
        code,
        ...(currentUserId && currentThreadId
          ? { originUserId: currentUserId, originThreadId: currentThreadId }
          : {}),
      });
    } catch (error) {
      console.error("Atlas runner window failed to open", error);
    } finally {
      window.setTimeout(() => setLaunching(false), 800);
    }
  };

  return (
    <div className="code-block-shell">
      <div className="code-block-header">
        <span>{formatLanguage(language)}</span>
        <div className="code-block-actions">
          {runnable ? <RunButton launching={launching} onRun={run} /> : null}
          <CopyButton copied={copied} onCopy={copy} />
        </div>
      </div>
      <SyntaxHighlighter
        PreTag="div"
        codeTagProps={{ className: "code-block-code" }}
        customStyle={{ margin: 0, background: "transparent", padding: "14px 16px" }}
        language={language}
        style={oneDark}
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

function formatLanguage(language: string) {
  if (!language) return "Code";
  const lower = language.toLowerCase();
  if (lower === "js") return "JavaScript";
  if (lower === "ts") return "TypeScript";
  if (lower === "tsx") return "TSX";
  if (lower === "jsx") return "JSX";
  if (lower === "py") return "Python";
  if (lower === "sh" || lower === "bash" || lower === "shell") return "Shell";
  if (lower === "json") return "JSON";
  if (lower === "yaml" || lower === "yml") return "YAML";
  if (lower === "md" || lower === "markdown") return "Markdown";
  if (lower === "sql") return "SQL";
  if (lower === "diff") return "Diff";
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function CopyButton({ copied, onCopy }: { copied: boolean; onCopy: () => Promise<void> }) {
  return (
    <button className="ghost-button compact-button code-copy-button" onClick={() => void onCopy()} type="button">
      {copied ? <Check size={14} /> : <Copy size={14} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function RunButton({ launching, onRun }: { launching: boolean; onRun: () => Promise<void> }) {
  return (
    <button
      className="ghost-button compact-button code-run-button"
      disabled={launching}
      onClick={() => void onRun()}
      type="button"
    >
      <Play size={14} />
      {launching ? "Opening..." : "Run"}
    </button>
  );
}

export { RUNNABLE_LANGUAGES };

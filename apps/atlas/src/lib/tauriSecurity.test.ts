import tauriConfig from "../../src-tauri/tauri.conf.json";
import indexHtml from "../../index.html?raw";
import { describe, expect, it } from "vitest";

const FRAME_SOURCES = ["'self'", "about:", "data:", "blob:", "http://127.0.0.1:*", "http://localhost:*"];

function directiveValue(csp: string, directive: string): string {
  const entry = csp
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${directive} `));
  return entry ?? "";
}

describe("Tauri security policy", () => {
  it("allows runner iframe previews without inline script execution", () => {
    const policies = [tauriConfig.app.security.csp, tauriConfig.app.security.devCsp];

    for (const policy of policies) {
      const frameSrc = directiveValue(policy, "frame-src");
      const scriptSrc = directiveValue(policy, "script-src");
      const scriptElementSrc = directiveValue(policy, "script-src-elem");
      const scriptAttributeSrc = directiveValue(policy, "script-src-attr");
      expect(frameSrc).not.toContain("'none'");
      for (const source of FRAME_SOURCES) {
        expect(frameSrc).toContain(source);
      }
      expect(policy).toContain("style-src-elem 'self' 'unsafe-inline'");
      expect(scriptSrc).toBe("script-src 'self'");
      expect(scriptElementSrc).toBe("script-src-elem 'self'");
      expect(scriptAttributeSrc).toBe("script-src-attr 'none'");
      expect(scriptSrc).not.toContain("'unsafe-inline'");
      expect(scriptElementSrc).not.toContain("'unsafe-inline'");
    }
  });

  it("loads the theme bootstrap as an external script", () => {
    expect(indexHtml).toContain('<script src="/theme-bootstrap.js"></script>');
    expect(indexHtml).not.toMatch(/<script>(?:(?!<\/script>)[\s\S])*localStorage/);
  });
});

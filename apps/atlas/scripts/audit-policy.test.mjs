import assert from "node:assert/strict";
import test from "node:test";

import { evaluateAuditReport } from "./audit-policy.mjs";

test("reports every high or critical vulnerability", () => {
  const report = {
    vulnerabilities: {
      "direct-package": { severity: "critical", via: [] },
      "indirect-package": { severity: "high", via: ["direct-package"] },
      "moderate-package": { severity: "moderate", via: [] },
    },
  };

  assert.deepEqual(evaluateAuditReport(report), [
    "direct-package: critical severity",
    "indirect-package: high severity",
  ]);
});

test("accepts a report with only lower-severity findings", () => {
  assert.deepEqual(
    evaluateAuditReport({
      vulnerabilities: {
        "moderate-package": { severity: "moderate", via: [] },
        "low-package": { severity: "low", via: [] },
      },
    }),
    [],
  );
});

test("does not exempt the former React Router advisory", () => {
  assert.deepEqual(
    evaluateAuditReport({
      vulnerabilities: {
        "react-router": {
          severity: "high",
          via: [
            {
              severity: "high",
              url: "https://github.com/advisories/GHSA-qwww-vcr4-c8h2",
            },
          ],
        },
        "react-router-dom": { severity: "high", via: ["react-router"] },
      },
    }),
    ["react-router: high severity", "react-router-dom: high severity"],
  );
});

test("fails closed for an unreadable report", () => {
  assert.throws(() => evaluateAuditReport(null), /unreadable report/);
});

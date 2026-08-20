import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { evaluateAuditReport } from "./audit-policy.mjs";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const audit = spawnSync(npmCommand, ["audit", "--json"], {
  cwd: appRoot,
  encoding: "utf8",
  maxBuffer: 20 * 1024 * 1024,
});
if (audit.error) {
  fail("npm audit could not start:", [audit.error.message]);
}

let report;
try {
  report = JSON.parse(audit.stdout);
} catch {
  fail("npm audit did not return valid JSON:", [
    audit.stderr.trim() || audit.stdout.trim() || `exit status ${audit.status}`,
  ]);
}

let blockers;
try {
  blockers = evaluateAuditReport(report);
} catch (error) {
  fail("npm audit policy could not validate the report:", [
    error instanceof Error ? error.message : String(error),
  ]);
}
if (blockers.length) {
  fail("Unapproved high or critical frontend vulnerabilities were found:", blockers);
}

console.log(
  "Frontend dependency audit passed with no high or critical advisories.",
);

function fail(message, details) {
  console.error(message);
  for (const detail of details) {
    console.error(`- ${detail}`);
  }
  process.exit(1);
}

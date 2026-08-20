const BLOCKING_SEVERITIES = new Set(["high", "critical"]);

export function evaluateAuditReport(report) {
  if (!report || typeof report !== "object" || !report.vulnerabilities) {
    throw new Error("npm audit returned an unreadable report.");
  }

  const vulnerabilities = report.vulnerabilities;
  const blockers = [];
  for (const [name, vulnerability] of Object.entries(vulnerabilities)) {
    if (!BLOCKING_SEVERITIES.has(String(vulnerability?.severity ?? "").toLowerCase())) {
      continue;
    }
    blockers.push(`${name}: ${vulnerability?.severity ?? "unknown"} severity`);
  }
  return blockers;
}

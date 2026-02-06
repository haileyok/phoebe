// Auto-generated - do not edit
import { callTool } from "./runtime.ts";

export const clickhouse = {
  /** Get Osprey/network table schema information including tables and their columns. Schema is for the table default.osprey_execution_results */
  getSchema: (): Promise<unknown> => callTool("clickhouse.getSchema", {}),

  /** Execute a SQL query against ClickHouse and return the results. All queries must include a LIMIT, and all queries must be executed on default.osprey_execution_results. */
  query: (sql: string): Promise<unknown> => callTool("clickhouse.query", { sql }),
};

export const domain = {
  /** Lookup A, AAAA, NS, MX, TXT, CNAME, and SOA for a given input domain */
  checkDomain: (domain: string): Promise<unknown> => callTool("domain.checkDomain", { domain }),
};

export const osprey = {
  /** Get Osprey configuration including available features, labels, and rules */
  getConfig: (): Promise<unknown> => callTool("osprey.getConfig", {}),

  /** Get available UDFs (user-defined functions) for rule writing */
  getUdfs: (): Promise<unknown> => callTool("osprey.getUdfs", {}),

  /** List existing .sml rule files in the ruleset. Use this before saving a rule to check for naming collisions. */
  listRuleFiles: (directory?: string): Promise<unknown> => callTool("osprey.listRuleFiles", { directory }),

  /** Read the contents of an existing .sml rule file in the ruleset. */
  readRuleFile: (file_path: string): Promise<unknown> => callTool("osprey.readRuleFile", { file_path }),

  /** Save an .sml rule file to the ruleset. Creates parent directories if needed. New files (except index.sml) are auto-registered in the parent directory's index.sml. Call osprey.listRuleFiles first to check for existing files. */
  saveRule: (file_path: string, content: string, require_if?: string): Promise<unknown> => callTool("osprey.saveRule", { file_path, content, require_if }),

  /** Validate the Osprey ruleset using the linter */
  validateRules: (): Promise<unknown> => callTool("osprey.validateRules", {}),
};

export const ozone = {
  /** Apply a moderation label to a subject (account or record) */
  applyLabel: (subject: string, label: string): Promise<unknown> => callTool("ozone.applyLabel", { subject, label }),

  /** Remove a moderation label from a subject (account or record) */
  removeLabel: (subject: string, label: string): Promise<unknown> => callTool("ozone.removeLabel", { subject, label }),
};

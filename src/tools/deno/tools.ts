// Auto-generated - do not edit
import { callTool } from "./runtime.ts";

export const clickhouse = {
  /** Get database schema information including tables and their columns. Schema is for the table default.osprey_execution_results */
  getSchema: (): Promise<unknown> => callTool("clickhouse.getSchema", {}),

  /** Execute a SQL query against ClickHouse and return the results. All queries must include a LIMIT, and all queries must be executed on default.osprey_execution_results. */
  query: (sql: string): Promise<unknown> =>
    callTool("clickhouse.query", { sql }),
};

export const osprey = {
  /** Get Osprey configuration including available features, labels, and rules */
  getConfig: (): Promise<unknown> => callTool("osprey.getConfig", {}),

  /** Get available UDFs (user-defined functions) for rule writing */
  getUdfs: (): Promise<unknown> => callTool("osprey.getUdfs", {}),
};

export const ozone = {
  /** Apply a moderation label to a subject (account or record) */
  applyLabel: (subject: string, label: string): Promise<unknown> =>
    callTool("ozone.applyLabel", { subject, label }),

  /** Remove a moderation label from a subject (account or record) */
  removeLabel: (subject: string, label: string): Promise<unknown> =>
    callTool("ozone.removeLabel", { subject, label }),
};

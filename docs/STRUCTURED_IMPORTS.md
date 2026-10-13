# Structured dataset imports

Planora supports CSV, TSV/delimited text, JSON/JSON Lines, Excel `.xlsx`, Apache Parquet `.parquet`, SQLite `.sqlite`/`.sqlite3`/`.db`, and ZIP archives containing these formats.

In Sources & settings choose **Upload Dataset**, select a file, choose a worksheet/table/archive member when prompted, then **Preview (no writes)** and **Commit load**. Import one selected table at a time. Use a distinct dataset name when importing different sheets or archive files into separate tables. Existing CSV/JSON behavior and the query-analysis/model pipeline are preserved.

## Format behavior

| Format | What is imported |
| --- | --- |
| Excel (.xlsx) | One selected worksheet. First row is headers. Formulas and error cells are rejected; export values first. Workbook formatting and charts are not imported. Legacy .xls/.xlsb and encrypted workbooks are not supported. |
| Parquet | One tabular dataset, read in batches. Nested lists/objects are serialized as JSON values. Binary columns are rejected. |
| SQLite | One ordinary table, opened read-only and immutable. Views, virtual tables, attached databases, SQL scripts and binary columns are not imported. Upload a complete database file after closing/checkpointing its writer; external WAL files are not read. |
| ZIP | Select one supported member, and then its sheet/table if needed. No extraction using member paths. Nested ZIP archives, encrypted members, duplicate member names, symlinks and unsafe paths are rejected. |
| TSV | Existing delimited-text detection supports tab-separated data. |

The new adapters feed the established profiling and PostgreSQL preview/commit pipeline. Types are inferred from values; original database types, keys, indexes, relationships, workbook styles and formulas are not preserved. Missing-value handling uses the existing configured null tokens. Date/time values are serialized as ISO text before inference. This is dataset import, not database backup restoration.

The configured upload limit applies to incoming files, expanded archives and converted output (100 MB by default). New structured adapters additionally cap conversion at one million rows and 512 columns; ZIP/XLSX containers are limited to 2,048 entries. Some compressed files therefore fail even when their uploaded size is small. JSON/CSV retain their existing limits.

Converted values are staged in temporary files and discarded through the existing preview lifecycle. Preview `sha256` and `size_bytes` refer to the normalized content for new formats; `filename` retains the original uploaded name. Different source files yielding identical normalized content use existing duplicate-load checks.

## API

- `POST /api/v1/data/options?filename=...`: raw file body; returns `members` and `tables`. For ZIP, repeat with `archive_member=...` to list its worksheet/table choices.
- `POST /api/v1/data/preview?filename=...&selection=...&archive_member=...`: select a sheet/table/member; existing dataset naming parameters remain available.
- `POST /api/v1/data/commit`: unchanged preview-token workflow.

These endpoints require the existing session header and service API key when configured. They do not add public user authentication; deployment remains subject to the separate public-release requirements.

Small synthetic fixtures are in `examples/structured/`. Each contains two sales rows (Delhi 12.5 and Pune 20.0). Models are unchanged.

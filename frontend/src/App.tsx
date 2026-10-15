import { useEffect, useMemo, useState, type FormEvent } from "react";
import type {
  BatchEstimate,
  ExtractionResult,
  FileLiveRow,
  HealthInfo,
  ReactionStep,
} from "./types";

function resultWarnings(r: ExtractionResult): string[] {
  const out: string[] = [];
  if (r.error) out.push(r.error);
  if (r.route?.warnings?.length) out.push(...r.route.warnings);
  for (const s of r.stages) {
    if (s.status === "error" && s.detail) out.push(`${s.label}: ${s.detail}`);
    if (s.id === "cache" && s.detail) out.push(s.detail);
  }
  return out.filter(Boolean);
}

function patchRow(
  rows: FileLiveRow[],
  filename: string,
  patch: Partial<FileLiveRow>,
): FileLiveRow[] {
  const idx = rows.findIndex((r) => r.filename === filename);
  if (idx < 0) {
    return [...rows, { filename, status: "queued", stageLabel: "", detail: "", warnings: [], ...patch }];
  }
  const next = [...rows];
  next[idx] = { ...next[idx], ...patch };
  return next;
}

function names(list: { name: string; amount?: string | null }[] | undefined): string {
  if (!list?.length) return "—";
  return list.map((x) => (x.amount && x.amount !== "N/A" ? `${x.name} (${x.amount})` : x.name)).join("; ");
}

function StepTable({ steps }: { steps: ReactionStep[] }) {
  if (!steps.length) {
    return <p className="muted">No reaction steps extracted.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Label</th>
            <th>Reactants</th>
            <th>Reagents</th>
            <th>Catalysts</th>
            <th>Solvents</th>
            <th>Products / intermediates</th>
            <th>Conditions</th>
            <th>Yield</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s) => (
            <tr key={s.step_index}>
              <td>{s.step_index}</td>
              <td>{s.step_label || "—"}</td>
              <td>{names(s.reactants)}</td>
              <td>{names(s.reagents)}</td>
              <td>{names(s.catalysts.concat(s.ligands || []))}</td>
              <td>{names(s.solvents)}</td>
              <td>
                {names(s.products)}
                {s.intermediates?.length ? (
                  <>
                    <br />
                    <span className="muted">int: {names(s.intermediates)}</span>
                  </>
                ) : null}
              </td>
              <td>
                {s.conditions?.temperature || "N/A"} / {s.conditions?.time || "N/A"}
                <br />
                <span className="muted">
                  {s.conditions?.atmosphere || "N/A"}
                  {s.conditions?.other && s.conditions.other !== "N/A" ? ` · ${s.conditions.other}` : ""}
                </span>
              </td>
              <td>
                {s.yield_value || "N/A"}
                {s.ee_or_dr && s.ee_or_dr !== "N/A" ? (
                  <>
                    <br />
                    <span className="muted">{s.ee_or_dr}</span>
                  </>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

async function readSseStream(
  res: Response,
  onEvent: (payload: Record<string, unknown>) => void,
): Promise<void> {
  if (!res.body) throw new Error("No response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .map((l) => l.trim())
        .find((l) => l.startsWith("data:"));
      if (!line) continue;
      const raw = line.replace(/^data:\s*/, "");
      if (!raw) continue;
      onEvent(JSON.parse(raw) as Record<string, unknown>);
    }
  }
}

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [estimating, setEstimating] = useState(false);
  const [estimate, setEstimate] = useState<BatchEstimate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [downloads, setDownloads] = useState<{ json: string; excel: string } | null>(null);
  const [progress, setProgress] = useState({
    index: 0,
    total: 0,
    filename: "",
    display: "",
    remaining: "",
    completed: 0,
    stageLabel: "",
    stageDetail: "",
    stages: [] as { id: string; label: string; status: string }[],
  });
  const [results, setResults] = useState<ExtractionResult[]>([]);
  const [fileRows, setFileRows] = useState<FileLiveRow[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [view, setView] = useState<"table" | "json">("table");
  const [finished, setFinished] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then(async (r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json() as Promise<HealthInfo>;
      })
      .then((data) => {
        setHealth(data);
        if (data.model_text) setModel(data.model_text);
      })
      .catch(() => setHealth(null));
  }, []);

  const onFilesChange = (list: FileList | null) => {
    if (!list) return;
    const pdfs = Array.from(list).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    setFiles(pdfs);
    setEstimate(null);
    setResults([]);
    setFileRows([]);
    setFinished(false);
    setDownloads(null);
    setBatchId(null);
  };

  const fetchEstimate = async () => {
    if (!files.length) return;
    setEstimating(true);
    setError(null);
    const body = new FormData();
    files.forEach((f) => body.append("files", f));
    try {
      const res = await fetch("/api/extract/batch/estimate", { method: "POST", body });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(
          text.includes("detail")
            ? (JSON.parse(text) as { detail?: string }).detail || `Estimate failed (${res.status})`
            : `Backend unavailable (${res.status}). Start FastAPI on :8765.`,
        );
      }
      const data = await res.json();
      setEstimate(data as BatchEstimate);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setEstimating(false);
    }
  };

  useEffect(() => {
    if (!files.length) {
      setEstimate(null);
      return;
    }
    const t = setTimeout(() => {
      void fetchEstimate();
    }, 400);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!files.length) {
      setError("Choose at least one PDF.");
      return;
    }
    setLoading(true);
    setFinished(false);
    setError(null);
    setResults([]);
    setFileRows(
      files.map((f) => {
        const estFile = estimate?.per_file.find((p) => p.filename === f.name);
        return {
          filename: f.name,
          status: "queued" as const,
          stageLabel: "Queued",
          detail: estFile?.cached
            ? "Will load from cache"
            : estFile?.estimated_display
              ? `Est. ${estFile.estimated_display}`
              : "Waiting…",
          pages: estFile?.page_count,
          figures: estFile?.figure_count,
          warnings: [],
          from_cache: estFile?.cached,
        };
      }),
    );
    setDownloads(null);
    setBatchId(null);
    setSelectedIdx(0);
    setProgress({
      index: 0,
      total: files.length,
      filename: "",
      display: estimate?.estimated_display ?? "",
      remaining: estimate?.estimated_display ?? "",
      completed: 0,
      stageLabel: "",
      stageDetail: "",
      stages: [],
    });

    const body = new FormData();
    files.forEach((f) => body.append("files", f));
    if (model.trim()) body.append("model", model.trim());

    try {
      const res = await fetch("/api/extract/batch/stream", { method: "POST", body });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Batch extraction failed");
      }

      let gotComplete = false;
      const collected: ExtractionResult[] = [];

      await readSseStream(res, (payload) => {
        const type = String(payload.type || "");
        if (type === "estimate") {
          setBatchId(String(payload.batch_id || ""));
          setProgress((p) => ({
            ...p,
            display: String(payload.estimated_display || p.display),
            total: Number(payload.file_count || p.total),
          }));
        } else if (type === "file_start") {
          const displayName =
            String(payload.display_filename || payload.filename || "") ||
            files[Number(payload.index || 1) - 1]?.name ||
            "";
          setProgress((p) => ({
            ...p,
            index: Number(payload.index || 0),
            total: Number(payload.total || files.length),
            filename: displayName,
            stageLabel: "Starting pipeline…",
            stageDetail: "",
            stages: [],
          }));
          setFileRows((rows) =>
            patchRow(rows, displayName, {
              status: "running",
              stageLabel: "Starting pipeline…",
              detail: "In progress",
            }),
          );
        } else if (type === "stage_update") {
          const fname = String(payload.filename || "");
          setProgress((p) => ({
            ...p,
            index: Number(payload.index || p.index),
            total: Number(payload.total || p.total),
            filename: fname || p.filename,
            stageLabel: String(payload.stage_label || p.stageLabel),
            stageDetail: String(payload.detail || p.stageDetail),
            remaining: String(payload.remaining_display || p.remaining),
            stages: (payload.stages as { id: string; label: string; status: string }[]) || p.stages,
          }));
          if (fname) {
            setFileRows((rows) =>
              patchRow(rows, fname, {
                status: "running",
                stageLabel: String(payload.stage_label || ""),
                detail: String(payload.detail || ""),
              }),
            );
          }
        } else if (type === "file_complete") {
          const result = payload.result as ExtractionResult;
          collected.push(result);
          setResults([...collected]);
          setProgress((p) => ({
            ...p,
            completed: collected.length,
            index: Number(payload.index || p.index),
            total: Number(payload.total || p.total),
            stageLabel: "File complete",
            stageDetail: `${result.route?.steps?.length ?? 0} step(s) extracted`,
          }));
          setFileRows((rows) =>
            patchRow(rows, result.filename, {
              status: result.from_cache ? "cached" : result.error ? "error" : "ok",
              stageLabel: result.from_cache ? "From cache" : "Complete",
              detail: result.error || `${result.route?.steps?.length ?? 0} step(s)`,
              pages: result.document?.page_count,
              figures: result.document?.figure_count,
              steps: result.route?.steps?.length ?? 0,
              warnings: resultWarnings(result),
              from_cache: result.from_cache,
              job_id: result.job_id,
            }),
          );
        } else if (type === "file_error") {
          if (payload.result) {
            const result = payload.result as ExtractionResult;
            collected.push(result);
            setResults([...collected]);
            setProgress((p) => ({
              ...p,
              completed: collected.length,
            }));
            setFileRows((rows) =>
              patchRow(rows, result.filename, {
                status: "error",
                stageLabel: "Failed",
                detail: String(payload.detail || result.error || "error"),
                warnings: resultWarnings(result),
                job_id: result.job_id,
              }),
            );
          } else {
            const fname = String(payload.filename || "");
            setFileRows((rows) =>
              patchRow(rows, fname, {
                status: "error",
                stageLabel: "Failed",
                detail: String(payload.detail || "error"),
                warnings: [String(payload.detail || "error")],
              }),
            );
          }
          setError((prev) =>
            prev
              ? `${prev}; ${payload.filename}: ${payload.detail}`
              : `${payload.filename}: ${payload.detail}`,
          );
        } else if (type === "batch_complete") {
          gotComplete = true;
          setBatchId(String(payload.batch_id || ""));
          setDownloads(payload.downloads as { json: string; excel: string });
          const finalResults = (payload.results as ExtractionResult[]) || collected;
          setResults(finalResults);
          setFileRows((rows) => {
            const byName = new Map(finalResults.map((r) => [r.filename, r]));
            return rows.map((row) => {
              const r = byName.get(row.filename);
              if (!r) return row;
              return {
                ...row,
                status: r.from_cache ? "cached" : r.error ? "error" : "ok",
                stageLabel: r.from_cache ? "From cache" : r.error ? "Failed" : "Complete",
                detail: r.error || `${r.route?.steps?.length ?? 0} step(s)`,
                pages: r.document?.page_count ?? row.pages,
                figures: r.document?.figure_count ?? row.figures,
                steps: r.route?.steps?.length ?? 0,
                warnings: resultWarnings(r),
                from_cache: r.from_cache,
                job_id: r.job_id,
              };
            });
          });
          setFinished(true);
        } else if (type === "error") {
          throw new Error(String(payload.detail || "Batch failed"));
        }
      });

      if (!gotComplete) throw new Error("Batch stream ended early.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setFinished(false);
    } finally {
      setLoading(false);
    }
  };

  const selected = results[selectedIdx] ?? null;
  const jsonText = useMemo(
    () => (selected ? JSON.stringify(selected.route ?? selected, null, 2) : ""),
    [selected],
  );
  const totalSteps = results.reduce((n, r) => n + (r.route?.steps?.length ?? 0), 0);
  const failedCount = results.filter((r) => r.error).length;
  const effectiveModel = model.trim() || health?.model_text || "(backend .env — start server to see model)";

  return (
    <div className="page">
      <header className="hero">
        <div className="brand">AVRSeek</div>
        <p className="tagline">Antiviral reaction knowledge from papers &amp; patents</p>
        <p className="sub">
          Upload one or more PDFs → batch extract → download combined JSON or Excel.
        </p>
      </header>

      <section className="panel status-panel">
        <div className="status-row">
          <span className={`dot ${health?.api_key_configured ? "ok" : "warn"}`} />
          <span>
            {health
              ? `${health.base_url} · model ${health.model_text}${
                  health.api_key_configured ? "" : " · API key missing in backend/.env"
                }`
              : "Backend offline — start FastAPI on :8765 (and Vite on :5173)"}
          </span>
        </div>
      </section>

      <section className="panel">
        <form className="upload-form" onSubmit={onSubmit}>
          <label className="field file-field">
            <span>Paper / patent PDFs (up to 20)</span>
            <input
              type="file"
              accept="application/pdf,.pdf"
              multiple
              disabled={loading}
              onChange={(e) => onFilesChange(e.target.files)}
            />
          </label>
          {files.length ? (
            <ul className="file-list">
              {files.map((f) => (
                <li key={f.name + f.size}>
                  {f.name} <span className="muted">({(f.size / 1024).toFixed(0)} KB)</span>
                </li>
              ))}
            </ul>
          ) : null}
          {estimate ? (
            <div className="estimate-box">
              <strong>Estimated processing time: {estimate.estimated_display}</strong>
              {estimate.cached_count ? (
                <p className="muted tiny">
                  {estimate.cached_count} file(s) already processed — will load from cache (skipped in estimate)
                </p>
              ) : null}
              <p className="muted tiny">
                {estimate.file_count} file(s) · {estimate.total_pages} pages · {estimate.total_figures}{" "}
                figures · {estimate.total_tables} table hints
              </p>
              {estimate.per_file.some((f) => f.cached) ? (
                <ul className="file-list cached-list">
                  {estimate.per_file
                    .filter((f) => f.cached)
                    .map((f) => (
                      <li key={f.filename}>
                        {f.filename} <span className="pill cached">cached</span>
                      </li>
                    ))}
                </ul>
              ) : null}
              <p className="muted tiny">{estimate.note}</p>
            </div>
          ) : estimating ? (
            <p className="muted">Estimating time…</p>
          ) : null}
          <label className="field">
            <span>LLM model (OpenAI-compatible)</span>
            <input
              type="text"
              value={model}
              disabled={loading}
              onChange={(e) => setModel(e.target.value)}
              placeholder={health?.model_text ?? "Leave empty → uses MODEL_TEXT from backend/.env"}
            />
            <p className="muted tiny">
              Will run as: <strong>{effectiveModel}</strong>
              {!health ? " · start the backend to sync from .env" : ""}
            </p>
          </label>
          <button type="submit" disabled={loading || !files.length}>
            {loading ? "Processing batch…" : `Run AVRSeek on ${files.length || 0} PDF(s)`}
          </button>
        </form>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {(loading || finished) && (
        <section className="panel">
          <h2>Batch progress</h2>
          {loading ? (
            <>
              <div className="progress-banner running" role="status">
                <span className="spinner" aria-hidden />
                <div>
                  <strong>
                    File {progress.index || 1} / {progress.total}
                    {progress.filename ? `: ${progress.filename}` : ""}
                  </strong>
                  {progress.stageLabel ? (
                    <p className="stage-current">
                      <span className="stage-current-label">{progress.stageLabel}</span>
                      {progress.stageDetail ? ` — ${progress.stageDetail}` : ""}
                    </p>
                  ) : null}
                  <p>
                    {progress.completed > 0 ? `${progress.completed} of ${progress.total} files done · ` : ""}
                    {progress.remaining
                      ? `~${progress.remaining} remaining`
                      : `Estimated total: ${progress.display || estimate?.estimated_display || "calculating…"}`}
                  </p>
                </div>
              </div>
              {progress.stages.length ? (
                <ol className="stage-checklist">
                  {progress.stages.map((s) => (
                    <li key={s.id} className={`stage-check ${s.status}`}>
                      <span className="stage-check-icon" aria-hidden />
                      <span className="stage-check-label">{s.label}</span>
                      <span className="stage-check-status">{s.status}</span>
                    </li>
                  ))}
                </ol>
              ) : null}
            </>
          ) : null}
          {fileRows.length ? (
            <div className="table-wrap live-summary">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>File</th>
                    <th>Status</th>
                    <th>Stage</th>
                    <th>Pages</th>
                    <th>Figures</th>
                    <th>Steps</th>
                    <th>Warnings</th>
                  </tr>
                </thead>
                <tbody>
                  {fileRows.map((row, i) => {
                    const resultIndex = results.findIndex((r) => r.filename === row.filename);
                    const clickable = resultIndex >= 0;
                    return (
                      <tr
                        key={row.filename}
                        className={`live-row ${row.status}${clickable && resultIndex === selectedIdx ? " selected" : ""}`}
                        onClick={() => {
                          if (clickable) setSelectedIdx(resultIndex);
                        }}
                        style={{ cursor: clickable ? "pointer" : "default" }}
                      >
                        <td>{i + 1}</td>
                        <td>
                          {row.filename}
                          {row.from_cache ? <span className="pill cached">cache</span> : null}
                        </td>
                        <td>
                          <span className={`badge ${row.status}`}>{row.status}</span>
                        </td>
                        <td>
                          {row.stageLabel}
                          {row.detail ? (
                            <>
                              <br />
                              <span className="muted">{row.detail}</span>
                            </>
                          ) : null}
                        </td>
                        <td>{row.pages ?? "—"}</td>
                        <td>{row.figures ?? "—"}</td>
                        <td>{row.steps ?? "—"}</td>
                        <td>
                          {row.warnings.length ? (
                            <ul className="warn-inline">
                              {row.warnings.map((w, wi) => (
                                <li key={wi}>{w}</li>
                              ))}
                            </ul>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {finished ? (
            <div className="progress-banner done" role="status">
              <strong>Batch finished</strong>
              <p>
                {results.length} file(s) · {totalSteps} step(s) total
                {failedCount ? ` · ${failedCount} failed` : ""}
                {batchId ? ` · batch ${batchId}` : ""}
              </p>
              {downloads ? (
                <div className="download-row">
                  <a className="btn-download" href={downloads.json} download>
                    Download JSON
                  </a>
                  <a className="btn-download" href={downloads.excel} download>
                    Download Excel
                  </a>
                </div>
              ) : null}
            </div>
          ) : null}
        </section>
      )}

      {results.length ? (
        <>
          <section className="panel">
            <h2>Results by file</h2>
            <div className="file-tabs">
              {results.map((r, i) => (
                <button
                  key={r.job_id}
                  type="button"
                  className={i === selectedIdx ? "active" : ""}
                  onClick={() => setSelectedIdx(i)}
                >
                  {r.filename}
                  {r.from_cache ? " ⚡" : r.error ? " ⚠" : ` (${r.route?.steps?.length ?? 0})`}
                </button>
              ))}
            </div>
            {selected ? (
              <>
                <ul className="stages">
                  {selected.stages.map((s) => (
                    <li key={s.id} className={`stage ${s.status}`}>
                      <div className="stage-top">
                        <strong>{s.label}</strong>
                        <span className="badge">{s.status}</span>
                      </div>
                      <p>{s.detail}</p>
                    </li>
                  ))}
                </ul>
                <div className="download-row">
                  <a className="btn-download secondary" href={`/api/jobs/${selected.job_id}/download.json`}>
                    JSON
                  </a>
                  <a className="btn-download secondary" href={`/api/jobs/${selected.job_id}/download.xlsx`}>
                    Excel
                  </a>
                </div>
              </>
            ) : null}
          </section>

          {selected ? (
            <section className="panel">
                <div className="panel-head">
                <h2>
                  {selected.filename}
                  {selected.from_cache ? <span className="pill cached">from cache</span> : null}
                </h2>
                <div className="tabs">
                  <button type="button" className={view === "table" ? "active" : ""} onClick={() => setView("table")}>
                    Table
                  </button>
                  <button type="button" className={view === "json" ? "active" : ""} onClick={() => setView("json")}>
                    JSON
                  </button>
                </div>
              </div>
              {selected.route ? (
                <div className="meta">
                  <div>
                    <span className="muted">Title</span>
                    <div>{selected.route.title || "—"}</div>
                  </div>
                  <div>
                    <span className="muted">Target</span>
                    <div>
                      {selected.route.target_name || "—"}{" "}
                      <span className="pill">{selected.route.target_class || "unknown"}</span>
                    </div>
                  </div>
                  <div>
                    <span className="muted">Doc type</span>
                    <div>{selected.route.document_type || "—"}</div>
                  </div>
                </div>
              ) : null}
              {view === "table" ? (
                <StepTable steps={selected.route?.steps ?? []} />
              ) : (
                <pre className="json">{jsonText}</pre>
              )}
              {selected.route?.warnings?.length ? (
                <div className="warnings">
                  <h3>Warnings</h3>
                  <ul>
                    {selected.route.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </section>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

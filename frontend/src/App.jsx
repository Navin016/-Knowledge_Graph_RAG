import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const PHASES = [
  ["upload", "PDF uploaded"],
  ["processing", "PDF processing + Gemini extraction"],
  ["normalization", "Relation normalization"],
  ["resolution", "Entity resolution"],
  ["neo4j_setup", "Neo4j setup"],
  ["neo4j_ingestion", "Neo4j graph + chunk ingestion"],
  ["embeddings", "Entity/vector embeddings"],
  ["verification", "Final Neo4j verification"],
  ["ready", "Ready for questions"],
];

function phaseMap(job) {
  const map = Object.fromEntries(PHASES.map(([id]) => [id, "pending"]));
  const phases = job?.phases ?? {};
  if (job?.status === "completed") {
    PHASES.forEach(([id]) => { map[id] = "completed"; });
    return map;
  }
  if (Array.isArray(phases)) {
    phases.forEach((p) => {
      if (p?.id) map[p.id] = p.status || "pending";
    });
  } else {
    Object.entries(phases).forEach(([id, value]) => {
      map[id] = typeof value === "string" ? value : (value?.status || "pending");
    });
  }
  return map;
}

function PhaseTracker({ job }) {
  const map = phaseMap(job);
  return (
    <div className="phase-list">
      {PHASES.map(([id, label], i) => (
        <React.Fragment key={id}>
          <div className={`phase ${map[id]}`}>
            <span className="phase-dot" />
            <div className="phase-main">
              <strong>{label}</strong>
              <small>
                {map[id] === "completed" ? "Completed" :
                 map[id] === "running" ? "In progress" :
                 map[id] === "failed" ? "Failed" : "Waiting"}
              </small>
            </div>
            <span className="phase-state">
              {map[id] === "completed" ? "Done" :
               map[id] === "running" ? "Running" :
               map[id] === "failed" ? "Error" : "Waiting"}
            </span>
          </div>
          {i < PHASES.length - 1 && <div className="phase-connector" />}
        </React.Fragment>
      ))}
    </div>
  );
}

function GraphView({ graph, onNodeClick }) {
  const svgRef = useRef(null);

  useEffect(() => {
    const svgNode = svgRef.current;
    if (!svgNode) return undefined;
    const svg = d3.select(svgNode);
    svg.selectAll("*").remove();

    const nodes = (graph?.nodes || []).map((n) => ({ ...n }));
    const links = (graph?.links || []).map((l) => ({ ...l }));
    if (!nodes.length) return undefined;

    const width = svgNode.clientWidth || 800;
    const height = svgNode.clientHeight || 560;
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g");
    const linkLayer = g.append("g");
    const nodeLayer = g.append("g");

    const link = linkLayer.selectAll("g").data(links).join("g");
    link.append("line");
    link.append("text").text((d) => d.relation || "");

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(125))
      .force("charge", d3.forceManyBody().strength(-440))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(42))
      .on("tick", () => {
        link.select("line")
          .attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
        link.select("text")
          .attr("x", (d) => (d.source.x + d.target.x) / 2)
          .attr("y", (d) => (d.source.y + d.target.y) / 2);
        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    const node = nodeLayer.selectAll("g").data(nodes).join("g")
      .on("click", (_, d) => onNodeClick?.(d))
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(.2).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

    node.append("circle").attr("r", 20);
    node.append("text").attr("dy", 34).text((d) => d.label || d.id);

    const zoom = d3.zoom().scaleExtent([.5, 2.5]).on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);

    return () => {
      simulation.stop();
      svg.on(".zoom", null);
    };
  }, [graph, onNodeClick]);

  if (!graph?.nodes?.length) {
    return <div className="graph-empty"><div><div className="graph-symbol">◇</div><strong>No graph yet</strong><p>Ask a question to explore retrieved entities and relationships.</p></div></div>;
  }
  return <div className="graph-wrap"><svg ref={svgRef} className="graph-svg" /></div>;
}

function SourceCard({ item, type, index }) {
  return (
    <article className="source-card">
      <div className="source-top">
        <span className="source-number">{index + 1}</span>
        <span className="source-type">{type}</span>
        <span className="source-score">
          {typeof item?.score === "number" ? `Score ${item.score.toFixed(4)}` : ""}
        </span>
      </div>
      <strong>{item?.chunk_id != null ? `Chunk ${item.chunk_id}` : "Graph path"}</strong>
      <p>{item?.text || "No source text returned."}</p>
      {item?.source_file && <small>{item.source_file}</small>}
    </article>
  );
}

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const ready = job?.status === "completed";
  const phases = useMemo(() => phaseMap(job), [job]);
  const completed = Object.values(phases).filter((v) => v === "completed").length;

  useEffect(() => {
    fetch(`${API_BASE}/api/current-document`)
      .then((r) => r.json())
      .then(async (d) => {
        if (!d?.ready || !d.job_id) return;
        const r = await fetch(`${API_BASE}/api/jobs/${d.job_id}`);
        const j = await r.json();
        if (r.ok) {
          setJob(j);
          setFile({ name: d.filename, size: 0 });
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!job?.job_id || !["queued", "running"].includes(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/jobs/${job.job_id}`);
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || "Could not read job status.");
        setJob(d);
        if (d.status === "failed") setError(d.error || "Ingestion failed.");
      } catch (e) {
        setError(e.message);
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [job?.job_id, job?.status]);

  async function upload(e) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setResponse(null);
    setQuestion("");
    setError("");
    setUploading(true);
    setPage("processing");
    try {
      const fd = new FormData();
      fd.append("file", selected);
      const r = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Upload failed.");
      setJob(d);
    } catch (e2) {
      setError(e2.message);
    } finally {
      setUploading(false);
    }
  }

  async function ask(e) {
    e.preventDefault();
    if (!ready || asking || !question.trim()) return;
    setAsking(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, job_id: job.job_id }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Question failed.");
      setResponse(d);
      setSubmittedQuestion(question.trim());
    } catch (e2) {
      setError(e2.message);
    } finally {
      setAsking(false);
    }
  }

  const vectorEvidence = response?.vector_evidence || response?.vector_sources || [];
  const graphEvidence = response?.graph_evidence || response?.graph_sources || [];

  return (
    <div className="app-shell">
      <div className="app-frame">
        <aside className="sidebar">
          <div className="brand-row">
            <div className="brand-name"><span className="brand-mark" />KG-RAG</div>
          </div>

          <nav className="nav">
            {[
              ["dashboard", "▦", "Dashboard"],
              ["processing", "◌", "Processing"],
              ["ask", "◍", "Ask"],
              ["evidence", "◈", "Evidence"],
            ].map(([id, icon, label]) => (
              <button
                key={id}
                className={`nav-item ${page === id ? "active" : ""}`}
                disabled={(id === "ask" && !ready) || (id === "evidence" && !response)}
                onClick={() => setPage(id)}
              >
                <span>{icon}</span>{label}
              </button>
            ))}
            <div className="nav-divider" />
           
          </nav>

          <div className="sidebar-doc">
            <div className="eyebrow">ACTIVE DOCUMENT</div>
            <strong>{file?.name || "No PDF loaded"}</strong>
            <small><span className={`mini-dot ${ready ? "green" : ""}`} />{ready ? "Ready for questions" : "Not ready"}</small>
          </div>

          <label className="sidebar-upload">
            <input type="file" accept="application/pdf" onChange={upload} disabled={uploading} />
            ＋ {uploading ? "Uploading…" : "Upload PDF"}
          </label>
        </aside>

        <main className="main">
          <header className="topbar">
            <div>
              <div className="eyebrow">KNOWLEDGE GRAPH + RAG</div>
              <h1>KG-RAG Explorer</h1>
              <p>Turn unstructured PDFs into searchable knowledge.</p>
            </div>
            <div className={`status-pill ${ready ? "ready" : ""}`}><span className="status-dot" />{ready ? "Ready" : job?.status === "running" ? "Processing" : "No document"}</div>
          </header>

          {page === "dashboard" && (
            <div className="page">
              <section className="hero">
                <div>
                  <div className="eyebrow">YOUR RESEARCH WORKSPACE</div>
                  <h2>Upload a paper.<br /><span>Turn it into knowledge.</span></h2>
               
                </div>
              </section>

              <section className="feature-grid">
                <article className="feature-card purple">
                  <div className="feature-icon">▤</div>
                  <h3>Turn Your PDF</h3>
                  <p>Extract entities, relationships, chunks, and retrieval data automatically.</p>
                  <label className="black-button"><input type="file" accept="application/pdf" onChange={upload} />Upload document</label>
                </article>
                <article className="feature-card blue">
                  <div className="feature-icon">✦</div>
                  <h3>Ask Your Knowledge Base</h3>
                  <p>Ask questions after the backend finishes the ingestion pipeline.</p>
                  <button className="black-button" disabled={!ready} onClick={() => setPage("ask")}>{ready ? "Start asking" : "Waiting for document"}</button>
                </article>
                <article className="feature-card pink">
                  <div className="feature-icon">◈</div>
                  <h3>Inspect The Evidence</h3>
                  <p>vector sources and graph retrieval sources.</p>
                  <button className="black-button" disabled={!response} onClick={() => setPage("evidence")}>View evidence</button>
                </article>
              </section>

              <section className="dashboard-grid">
                <div className="panel processing-card">
                  <div className="section-head"><div><div className="panel-title">Processing status</div><div className="panel-subtitle">Live backend job status.</div></div><button className="text-button" onClick={() => setPage("processing")}>Open →</button></div>
                  <div className="processing-overview">
                    <div className="percent">{job?.progress ?? 0}<small>%</small></div>
                    <div><strong>{file?.name || "No active PDF"}</strong><p>{job?.message || "Upload a PDF to begin."}</p></div>
                  </div>
                  <div className="tiny-phase-row">{PHASES.map(([id]) => <span key={id} className={`tiny-phase ${phases[id]}`} />)}</div>
                </div>
                <div className="panel document-summary">
                  <div className="section-head"><div><div className="panel-title">Current document</div><div className="panel-subtitle">Active source for questions.</div></div></div>
                  <div className="document-body"><div className="pdf-badge">PDF</div><div><strong>{file?.name || "No PDF loaded"}</strong><p>{ready ? "Processed and ready" : "Awaiting upload"}</p></div><span className={`document-dot ${ready ? "green" : ""}`} /></div>
                </div>
              </section>
            </div>
          )}

          {page === "processing" && (
            <div className="page narrow">
              <div className="page-heading"><div><div className="eyebrow">DOCUMENT PIPELINE</div><h2>Processing your document</h2><p>Every stage is driven by backend status.</p></div></div>
              <section className="panel process-hero">
                <div className="process-file"><div className="pdf-badge large">PDF</div><div><strong>{file?.name || "No document"}</strong><p>{job?.message || "No processing job."}</p></div><b>{job?.progress ?? 0}%</b></div>
                <div className="big-track"><div style={{ width: `${job?.progress ?? 0}%` }} /></div>
              </section>
              <section className="panel phase-panel">
                <div className="section-head"><div><div className="panel-title">Pipeline progress</div><div className="panel-subtitle">{completed} of {PHASES.length} stages completed.</div></div><span className="live"><i />Live</span></div>
                <PhaseTracker job={job} />
              </section>
              {ready && <button className="black-button wide" onClick={() => setPage("ask")}>Continue to questions →</button>}
            </div>
          )}

          {page === "ask" && (
            <div className="page">
              <div className="page-heading"><div><div className="eyebrow">ASK YOUR DOCUMENT</div><h2>What do you want to know?</h2><p>Answers are generated from the active document after ingestion completes.</p></div></div>
              <section className="ask-layout">
                <div className="panel ask-card">
                  <div className="card-head"><div><div className="panel-title">Ask your document</div><div className="panel-subtitle">Ask a question to run vector retrieval, graph retrieval, and Gemini synthesis.</div></div><span className="ready-badge"><i />Ready</span></div>
                  <form onSubmit={ask} className="question-form">
                    <textarea value={question} onChange={(e) => setQuestion(e.target.value)} disabled={!ready || asking} placeholder="Ask something about the document…" rows={6} />
                    <div className="question-actions"><small>Grounded in the active PDF</small><button className="black-button" disabled={!ready || asking || !question.trim()}>{asking ? "Thinking…" : "Ask question"} <span>→</span></button></div>
                  </form>
                  {response ? <div className="answer-area"><div className="eyebrow">ANSWER</div><h3>{submittedQuestion}</h3><div className="answer-text">{response.answer}</div><button className="outline-button" onClick={() => setPage("evidence")}>View evidence →</button></div> : <div className="ask-empty"><div>✦</div><strong>Your answer will appear here</strong><p>Ask a question to run vector retrieval, graph retrieval, and Gemini synthesis.</p></div>}
                </div>
                <div className="panel graph-card">
                  <div className="card-head"><div><div className="panel-title">Retrieved knowledge graph</div><div className="panel-subtitle">Click an entity to inspect it.</div></div>{response?.graph && <small>{response.graph.nodes?.length || 0} nodes · {response.graph.links?.length || 0} links</small>}</div>
                  <GraphView graph={response?.graph} onNodeClick={setSelectedNode} />
                  {selectedNode && <div className="node-popover"><div className="eyebrow">ENTITY</div><strong>{selectedNode.label}</strong><button className="text-button" onClick={() => setSelectedNode(null)}>Close</button></div>}
                </div>
              </section>
            </div>
          )}

          {page === "evidence" && (
            <div className="page">
              <div className="page-heading"><div><div className="eyebrow">RETRIEVAL EVIDENCE</div><h2>Why the answer looks like this</h2><p>Vector and graph evidence are separated from the answer view.</p></div></div>
              <section className="panel evidence-summary">
                <div><div className="eyebrow">QUESTION</div><strong>{submittedQuestion}</strong></div>
                <div className="metrics"><div><b>{vectorEvidence.length}</b><span>Vector sources</span></div><div><b>{graphEvidence.length}</b><span>Graph paths</span></div><div><b>{response?.graph?.nodes?.length || 0}</b><span>Graph entities</span></div></div>
              </section>
              <div className="evidence-grid">
                <section><div className="column-head"><div><div className="panel-title">Vector retrieval</div><div className="panel-subtitle">Text chunks selected by semantic similarity.</div></div><b>{vectorEvidence.length}</b></div>{vectorEvidence.length ? vectorEvidence.map((x, i) => <SourceCard key={i} item={x} type="Vector" index={i} />) : <div className="empty-box">No vector evidence returned.</div>}</section>
                <section><div className="column-head"><div><div className="panel-title">Graph retrieval</div><div className="panel-subtitle">Entity paths selected from Neo4j.</div></div><b>{graphEvidence.length}</b></div>{graphEvidence.length ? graphEvidence.map((x, i) => <SourceCard key={i} item={x} type="Graph" index={i} />) : <div className="empty-box">No graph evidence returned.</div>}</section>
              </div>
            </div>
          )}
        </main>
      </div>

      {error && <div className="error-toast"><strong>Something went wrong</strong><div>{error}</div><button onClick={() => setError("")}>Dismiss</button></div>}
    </div>
  );
}

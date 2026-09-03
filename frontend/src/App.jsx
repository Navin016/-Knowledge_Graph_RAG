import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function GraphView({ graph }) {
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const nodes = graph?.nodes || [];
    const links = graph?.links || [];

    if (!nodes.length) {
      return undefined;
    }

    const width = svgRef.current.clientWidth || 700;
    const height = svgRef.current.clientHeight || 520;

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "currentColor");

    const linkLayer = svg.append("g");
    const nodeLayer = svg.append("g");

    const link = linkLayer
      .selectAll("g")
      .data(links)
      .join("g")
      .attr("class", "graph-link")
      .attr("marker-end", "url(#arrow)");

    link.append("line");
    link
      .append("text")
      .attr("class", "link-label")
      .text((d) => d.relation);

    const node = nodeLayer
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", "graph-node")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    node.append("circle").attr("r", 20);
    node
      .append("text")
      .attr("class", "node-label")
      .attr("dy", 34)
      .text((d) => d.label);

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance(120)
      )
      .force("charge", d3.forceManyBody().strength(-450))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(34))
      .on("tick", () => {
        link
          .select("line")
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);

        link
          .select("text")
          .attr("x", (d) => (d.source.x + d.target.x) / 2)
          .attr("y", (d) => (d.source.y + d.target.y) / 2);

        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

    return () => simulation.stop();
  }, [graph]);

  return (
    <div className="graph-wrap">
      {graph?.nodes?.length ? (
        <svg ref={svgRef} className="graph-svg" />
      ) : (
        <div className="graph-empty">
          Ask a question to see its retrieved knowledge graph.
        </div>
      )}
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [job, setJob] = useState(null);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const ready = job?.status === "completed";

  const statusLabel = useMemo(() => {
    if (!job) return "No PDF loaded";
    if (job.status === "queued") return "Queued";
    if (job.status === "running") return "Processing";
    if (job.status === "completed") return "Ready";
    return "Failed";
  }, [job]);

  useEffect(() => {
    if (!job?.job_id || !["queued", "running"].includes(job.status)) {
      return undefined;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${job.job_id}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Could not read ingestion status.");
        setJob(data);
        if (data.status === "failed") {
          setError(data.error || "PDF ingestion failed.");
        }
      } catch (err) {
        setError(err.message);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [job?.job_id, job?.status]);

  async function handleUpload(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setResponse(null);
    setError("");
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", selected);

      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed.");

      setJob(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    if (!question.trim() || !ready || asking) return;

    setAsking(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          job_id: job.job_id,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Question failed.");

      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">KNOWLEDGE GRAPH + RAG</div>
          <h1>KG-RAG Explorer</h1>
          <p>Upload a technical PDF, ask questions, and inspect the retrieved graph.</p>
        </div>
        <div className={`status-pill ${ready ? "ready" : ""}`}>
          <span className="status-dot" /> {statusLabel}
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <section className="panel upload-panel">
            <div className="panel-title">Document</div>
            <label className="upload-dropzone">
              <input type="file" accept="application/pdf" onChange={handleUpload} disabled={uploading} />
              <span className="upload-icon">↥</span>
              <strong>{uploading ? "Uploading..." : "Choose PDF"}</strong>
              <small>PDF only · max 50 MB</small>
            </label>

            {file && (
              <div className="file-card">
                <div className="file-name">{file.name}</div>
                <div className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
              </div>
            )}

            {job && (
              <div className="ingestion-box">
                <div className="ingestion-header">
                  <span>{statusLabel}</span>
                  <span>{job.progress ?? 0}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${job.progress ?? 0}%` }} />
                </div>
                <p>{job.message}</p>
              </div>
            )}
          </section>

          <section className="panel architecture-panel">
            <div className="panel-title">Runtime flow</div>
            <div className="flow-step">Question</div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">Vector retrieval + Graph retrieval</div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">Gemini answer synthesis</div>
            <div className="flow-arrow">↓</div>
            <div className="flow-step">Answer + D3 subgraph</div>
          </section>
        </aside>

        <section className="content">
          <div className="content-grid">
            <section className="panel chat-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-title">Ask your document</div>
                  <div className="panel-subtitle">
                    Questions are answered from the currently loaded PDF.
                  </div>
                </div>
              </div>

              <form className="question-form" onSubmit={askQuestion}>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder={ready ? "What does the document say about..." : "Upload and finish processing a PDF first"}
                  disabled={!ready || asking}
                  rows={3}
                />
                <button type="submit" disabled={!ready || asking || !question.trim()}>
                  {asking ? "Thinking..." : "Ask question"}
                </button>
              </form>

              {response ? (
                <div className="answer-area">
                  <div className="answer-card">
                    <div className="answer-label">ANSWER</div>
                    <div className="answer-text">{response.answer}</div>
                  </div>

                  <div className="evidence-grid">
                    <div>
                      <div className="section-label">Vector evidence</div>
                      {response.vector_evidence.map((item, index) => (
                        <div className="evidence-card" key={`${item.chunk_id}-${index}`}>
                          <div className="evidence-meta">
                            Chunk {item.chunk_id} · score {item.score.toFixed(4)}
                          </div>
                          <div>{item.text}</div>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div className="section-label">Graph evidence</div>
                      {response.graph_evidence.map((item, index) => (
                        <div className="evidence-card" key={`${item.text}-${index}`}>
                          <div className="evidence-meta">
                            path score {item.score.toFixed(4)}
                          </div>
                          <div>{item.text}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-title">Your answer will appear here</div>
                  <p>Once the document is ready, ask a question to run the existing KG-RAG retrieval pipeline.</p>
                </div>
              )}
            </section>

            <section className="panel graph-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-title">Retrieved knowledge graph</div>
                  <div className="panel-subtitle">
                    Drag nodes to explore entities and relationships selected for the question.
                  </div>
                </div>
                {response?.graph && (
                  <div className="graph-count">
                    {response.graph.nodes.length} nodes · {response.graph.links.length} links
                  </div>
                )}
              </div>
              <GraphView graph={response?.graph} />
            </section>
          </div>
        </section>
      </main>

      {error && <div className="error-toast">{error}</div>}
    </div>
  );
}

export default App;

import { useState, useCallback, useRef } from "react";

const API_BASE = "/api";

const MODULE_META = {
  brain_mri:   { label: "Brain MRI",           icon: "🧠", desc: "Glioma · Meningioma · Pituitary · No Tumor" },
  chest_xray:  { label: "Chest X-Ray",          icon: "🫁", desc: "Pneumonia · COVID-19 · Tuberculosis · Normal" },
  skin_lesion: { label: "Skin Lesion",           icon: "🔬", desc: "Melanoma · Nevus · Carcinoma · 4 more" },
  retinal:     { label: "Diabetic Retinopathy",  icon: "👁", desc: "5-stage DR grading" },
};

const SEVERITY = {
  "Glioma": "high", "Meningioma": "high", "Pituitary": "medium",
  "No Tumor": "low", "Normal": "low", "Pneumonia": "high",
  "COVID-19": "high", "Tuberculosis": "high", "Melanoma": "high",
  "No DR": "low", "Mild DR": "low", "Moderate DR": "medium",
  "Severe DR": "high", "Proliferative DR": "high",
};

const severityColor = (s) => ({
  high:   "#ef4444",
  medium: "#f59e0b",
  low:    "#22c55e",
}[s] || "#64748b");

export default function App() {
  const [module, setModule]       = useState("brain_mri");
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef                  = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("module_key", module);
      const res  = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(null); };

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logo}>
            <span style={styles.logoMark}>⬡</span>
            <span style={styles.logoText}>MedScan <span style={styles.logoAI}>AI</span></span>
          </div>
          <span style={styles.headerTag}>EfficientNetV2 · Grad-CAM++ · 99.8% AUC</span>
        </div>
      </header>

      <main style={styles.main}>
        {/* ── Module selector ── */}
        <section style={styles.section}>
          <p style={styles.sectionLabel}>SELECT MODULE</p>
          <div style={styles.moduleGrid}>
            {Object.entries(MODULE_META).map(([key, meta]) => (
              <button
                key={key}
                style={{ ...styles.moduleCard, ...(module === key ? styles.moduleCardActive : {}) }}
                onClick={() => { setModule(key); reset(); }}
              >
                <span style={styles.moduleIcon}>{meta.icon}</span>
                <span style={styles.moduleLabel}>{meta.label}</span>
                <span style={styles.moduleDesc}>{meta.desc}</span>
              </button>
            ))}
          </div>
        </section>

        {/* ── Upload + Result ── */}
        <div style={styles.workArea}>
          {/* Upload panel */}
          <section style={styles.uploadPanel}>
            <p style={styles.sectionLabel}>UPLOAD IMAGE</p>
            <div
              style={{ ...styles.dropzone, ...(dragging ? styles.dropzoneDragging : {}) }}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => !preview && inputRef.current.click()}
            >
              {preview ? (
                <div style={styles.previewWrap}>
                  <img src={preview} alt="preview" style={styles.previewImg} />
                  <button style={styles.clearBtn} onClick={(e) => { e.stopPropagation(); reset(); }}>✕ Clear</button>
                </div>
              ) : (
                <div style={styles.dropPrompt}>
                  <span style={styles.dropIcon}>⬆</span>
                  <p style={styles.dropText}>Drop image here or <span style={styles.dropLink}>browse</span></p>
                  <p style={styles.dropSub}>JPEG · PNG · up to 10MB</p>
                </div>
              )}
              <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])} />
            </div>

            <button
              style={{ ...styles.analyzeBtn, ...((!file || loading) ? styles.analyzeBtnDisabled : {}) }}
              onClick={predict}
              disabled={!file || loading}
            >
              {loading ? <><Spinner /> Analyzing…</> : "Analyze Image →"}
            </button>

            {error && <div style={styles.errorBox}>⚠ {error}</div>}
          </section>

          {/* Result panel */}
          <section style={styles.resultPanel}>
            <p style={styles.sectionLabel}>ANALYSIS RESULT</p>
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div style={styles.resultEmpty}>
                {loading
                  ? <LoadingState />
                  : <EmptyState module={MODULE_META[module]} />}
              </div>
            )}
          </section>
        </div>
      </main>

      <footer style={styles.footer}>
        For research and educational use only · Not a substitute for clinical diagnosis
      </footer>
    </div>
  );
}

function ResultCard({ result }) {
  const sev   = SEVERITY[result.predicted_class] || "medium";
  const color = severityColor(sev);
  const probs = Object.entries(result.class_probs).sort((a, b) => b[1] - a[1]);

  return (
    <div style={styles.resultCard}>
      {/* Prediction badge */}
      <div style={{ ...styles.predBadge, borderColor: color }}>
        <span style={{ ...styles.predLabel, color }}>
          {sev === "low" ? "✓ " : sev === "high" ? "⚠ " : "● "}
          {result.predicted_class}
        </span>
        <span style={styles.predConf}>{result.confidence}% confidence</span>
      </div>

      {/* Confidence bars */}
      <div style={styles.barsWrap}>
        {probs.map(([cls, prob]) => (
          <div key={cls} style={styles.barRow}>
            <span style={styles.barLabel}>{cls}</span>
            <div style={styles.barTrack}>
              <div style={{
                ...styles.barFill,
                width: `${prob}%`,
                background: cls === result.predicted_class ? color : "#334155",
              }} />
            </div>
            <span style={styles.barPct}>{prob.toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* Grad-CAM overlay */}
      {result.overlay_b64 && (
        <div style={styles.gradcamWrap}>
          <p style={styles.gradcamLabel}>GRAD-CAM++ ACTIVATION MAP</p>
          <img
            src={`data:image/png;base64,${result.overlay_b64}`}
            alt="Grad-CAM"
            style={styles.gradcamImg}
          />
          <p style={styles.gradcamSub}>Red regions indicate areas of highest diagnostic significance</p>
        </div>
      )}

      <div style={styles.metaRow}>
        <span style={styles.metaChip}>{result.module_name}</span>
        <span style={styles.metaChip}>{result.inference_ms}ms</span>
        <span style={styles.metaChip}>EfficientNetV2-S</span>
      </div>
    </div>
  );
}

function EmptyState({ module }) {
  return (
    <div style={styles.emptyWrap}>
      <span style={{ fontSize: 48 }}>{module.icon}</span>
      <p style={styles.emptyTitle}>{module.label}</p>
      <p style={styles.emptyDesc}>{module.desc}</p>
      <p style={styles.emptyHint}>Upload an image and click Analyze</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div style={styles.emptyWrap}>
      <div style={styles.loadingRing} />
      <p style={styles.emptyTitle}>Running inference…</p>
      <p style={styles.emptyDesc}>EfficientNetV2 · Grad-CAM++ · MPS</p>
    </div>
  );
}

function Spinner() {
  return <span style={styles.spinner}>⟳ </span>;
}

// ── Styles ────────────────────────────────────
const styles = {
  root: {
    minHeight: "100vh",
    background: "#050d1a",
    color: "#e2e8f0",
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    borderBottom: "1px solid #0f2340",
    background: "rgba(5,13,26,0.95)",
    backdropFilter: "blur(10px)",
    position: "sticky",
    top: 0,
    zIndex: 100,
  },
  headerInner: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: "16px 24px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  logo: { display: "flex", alignItems: "center", gap: 10 },
  logoMark: { fontSize: 22, color: "#38bdf8" },
  logoText: { fontSize: 18, fontWeight: 700, letterSpacing: "-0.5px", color: "#f1f5f9" },
  logoAI: { color: "#38bdf8" },
  headerTag: { fontSize: 11, color: "#475569", letterSpacing: "0.08em", fontFamily: "monospace" },

  main: { flex: 1, maxWidth: 1100, margin: "0 auto", padding: "40px 24px", width: "100%" },

  section: { marginBottom: 36 },
  sectionLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: "0.15em",
    color: "#38bdf8", marginBottom: 12, fontFamily: "monospace",
  },

  moduleGrid: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 },
  moduleCard: {
    background: "#0a1628",
    border: "1px solid #0f2340",
    borderRadius: 10,
    padding: "16px 14px",
    cursor: "pointer",
    textAlign: "left",
    display: "flex",
    flexDirection: "column",
    gap: 4,
    transition: "all 0.15s",
    color: "#94a3b8",
  },
  moduleCardActive: {
    border: "1px solid #38bdf8",
    background: "#0c1f3d",
    color: "#f1f5f9",
    boxShadow: "0 0 20px rgba(56,189,248,0.1)",
  },
  moduleIcon: { fontSize: 22, marginBottom: 4 },
  moduleLabel: { fontSize: 13, fontWeight: 600, color: "inherit" },
  moduleDesc: { fontSize: 10, color: "#475569", lineHeight: 1.4 },

  workArea: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 },

  uploadPanel: { display: "flex", flexDirection: "column", gap: 14 },
  dropzone: {
    border: "2px dashed #0f2340",
    borderRadius: 12,
    minHeight: 280,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    transition: "all 0.15s",
    background: "#0a1628",
    overflow: "hidden",
  },
  dropzoneDragging: { border: "2px dashed #38bdf8", background: "#0c1f3d" },
  dropPrompt: { textAlign: "center", padding: 32 },
  dropIcon: { fontSize: 32, color: "#1e3a5f", display: "block", marginBottom: 12 },
  dropText: { fontSize: 14, color: "#64748b", marginBottom: 6 },
  dropLink: { color: "#38bdf8", textDecoration: "underline" },
  dropSub: { fontSize: 11, color: "#334155", fontFamily: "monospace" },

  previewWrap: { position: "relative", width: "100%", height: "100%" },
  previewImg: { width: "100%", height: 280, objectFit: "contain", background: "#050d1a" },
  clearBtn: {
    position: "absolute", top: 8, right: 8,
    background: "rgba(0,0,0,0.7)", color: "#94a3b8",
    border: "1px solid #1e3a5f", borderRadius: 6,
    padding: "4px 10px", fontSize: 11, cursor: "pointer",
  },

  analyzeBtn: {
    background: "#38bdf8",
    color: "#050d1a",
    border: "none",
    borderRadius: 8,
    padding: "14px 24px",
    fontSize: 14,
    fontWeight: 700,
    cursor: "pointer",
    letterSpacing: "0.02em",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    transition: "all 0.15s",
  },
  analyzeBtnDisabled: { background: "#0f2340", color: "#334155", cursor: "not-allowed" },

  errorBox: {
    background: "#1a0a0a", border: "1px solid #7f1d1d",
    borderRadius: 8, padding: "10px 14px",
    fontSize: 12, color: "#fca5a5",
  },

  resultPanel: {},
  resultEmpty: {
    border: "1px solid #0f2340",
    borderRadius: 12,
    minHeight: 340,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0a1628",
  },

  resultCard: { display: "flex", flexDirection: "column", gap: 20 },
  predBadge: {
    border: "1px solid",
    borderRadius: 10,
    padding: "16px 20px",
    background: "#0a1628",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  predLabel: { fontSize: 20, fontWeight: 700, letterSpacing: "-0.3px" },
  predConf: { fontSize: 13, color: "#64748b", fontFamily: "monospace" },

  barsWrap: {
    background: "#0a1628",
    border: "1px solid #0f2340",
    borderRadius: 10,
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  barRow: { display: "flex", alignItems: "center", gap: 10 },
  barLabel: { fontSize: 11, color: "#64748b", width: 110, flexShrink: 0 },
  barTrack: { flex: 1, height: 6, background: "#0f2340", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 3, transition: "width 0.5s ease" },
  barPct: { fontSize: 11, color: "#475569", fontFamily: "monospace", width: 40, textAlign: "right" },

  gradcamWrap: {
    background: "#0a1628",
    border: "1px solid #0f2340",
    borderRadius: 10,
    overflow: "hidden",
  },
  gradcamLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: "0.15em",
    color: "#38bdf8", padding: "12px 16px 8px", fontFamily: "monospace",
  },
  gradcamImg: { width: "100%", display: "block" },
  gradcamSub: {
    fontSize: 10, color: "#475569", padding: "8px 16px 12px",
    fontFamily: "monospace", textAlign: "center",
  },

  metaRow: { display: "flex", gap: 8, flexWrap: "wrap" },
  metaChip: {
    fontSize: 10, fontFamily: "monospace",
    background: "#0a1628", border: "1px solid #0f2340",
    borderRadius: 4, padding: "3px 8px", color: "#475569",
  },

  emptyWrap: { textAlign: "center", padding: 32, display: "flex", flexDirection: "column", alignItems: "center", gap: 8 },
  emptyTitle: { fontSize: 16, fontWeight: 600, color: "#475569", marginTop: 8 },
  emptyDesc: { fontSize: 12, color: "#334155" },
  emptyHint: { fontSize: 11, color: "#1e3a5f", marginTop: 4, fontFamily: "monospace" },

  loadingRing: {
    width: 40, height: 40,
    border: "3px solid #0f2340",
    borderTop: "3px solid #38bdf8",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  spinner: { display: "inline-block", animation: "spin 0.8s linear infinite" },

  footer: {
    textAlign: "center",
    padding: "16px 24px",
    fontSize: 10,
    color: "#1e3a5f",
    fontFamily: "monospace",
    letterSpacing: "0.05em",
    borderTop: "1px solid #0a1628",
  },
};

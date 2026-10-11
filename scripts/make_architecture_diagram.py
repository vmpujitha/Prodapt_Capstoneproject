import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        8, 9.6, "AI-Powered Agriculture Crop Advisory Assistant Architecture",
        fontsize=18, fontweight="bold", ha="center", color="#0f172a"
    )
    ax.text(
        8, 9.25, "Multi-Agent LangGraph Orchestration • Hybrid Retrieval • Classical ML • LLM-as-Judge",
        fontsize=11, color="#475569", ha="center"
    )

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, text_color="#0f172a"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, fontsize=10.5, fontweight="bold", ha="center", va="center", color=text_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h*0.32, subtitle, fontsize=8.5, ha="center", va="center", color="#475569", zorder=3)

    # Layer 1: Data Ingestion & Indexing
    draw_box(0.5, 7.2, 3.2, 1.4, "Kaggle Crop/Soil/Weather CSVs", "Merged ~19.6k cases (synthetic fallback)\nCanonical schema mapping + cleaning", "#ecfdf5", "#10b981", "#065f46")
    draw_box(4.2, 7.2, 3.2, 1.4, "Data Processing & Feat. Eng", "Tertile Cutoffs (thresholds.json)\nNarrative Builder (CASE-XXXXXX)", "#ecfdf5", "#10b981", "#065f46")
    draw_box(7.9, 7.2, 3.4, 1.4, "Dual Storage & Vector Index", "ChromaDB (cases__provider__model)\nBM25 Okapi Index (Cache)", "#ecfdf5", "#10b981", "#065f46")
    draw_box(11.8, 7.2, 3.7, 1.4, "ML Model Training", "Yield: GradientBoosting/XGBoost\nRisk: Calibrated Classifier", "#ecfdf5", "#10b981", "#065f46")

    # Layer 2: User Interaction & API Gateway
    draw_box(0.5, 5.0, 3.2, 1.5, "React + Vite UI", "Advisor, Compare, Eval, Feedback\nProvider & Mode Selector", "#eff6ff", "#3b82f6", "#1e40af")
    draw_box(4.2, 5.0, 3.2, 1.5, "FastAPI Service", "REST Endpoints (/api/v1/query)\nCORS, Rate Limit, Lifespan", "#eff6ff", "#3b82f6", "#1e40af")
    draw_box(7.9, 5.0, 3.4, 1.5, "Input Guardrails", "Length, SQL/Injection, Domain Check\nPhysical Bounds (pH, NPK, Weather)", "#fef2f2", "#ef4444", "#991b1b")
    draw_box(11.8, 5.0, 3.7, 1.5, "Classification Agent", "Intent, Crop, Region, Season\nRegex / Deterministic Fallback", "#fdf4ff", "#d946ef", "#86198f")

    # Layer 3: LangGraph Multi-Agent Collaboration
    draw_box(0.5, 2.6, 3.2, 1.7, "Hybrid Retrieval (RRF, k=60)", "BM25 + Chroma, shared metadata filters\nPrioritization: proximity, yield, recency,\nfeedback boost", "#fffbeb", "#f59e0b", "#92400e")
    draw_box(4.2, 2.6, 3.2, 1.7, "Parallel Analysis Agents", "• Impact Analyst (What-If Sens.)\n• Yield Model (Intervals)\n• Risk Model (Suitability)", "#fffbeb", "#f59e0b", "#92400e")
    draw_box(7.9, 2.6, 3.4, 1.7, "Recommender Agent", "Synthesis of Candidates\nGrounding in Retrieved Cases", "#fffbeb", "#f59e0b", "#92400e")
    draw_box(11.8, 2.6, 3.7, 1.7, "Output Guardrails + LLM Judge", "Case-ID & number verification\nPass/fail computed in code (no rewrite loop)", "#fef2f2", "#ef4444", "#991b1b")

    # Layer 4: Foundation LLM Providers & Fallback Engine
    draw_box(0.5, 0.5, 3.2, 1.4, "1. Ollama (default)", "llama3.1:8b chat\nnomic-embed-text", "#f1f5f9", "#64748b", "#334155")
    draw_box(4.2, 0.5, 3.2, 1.4, "2. OpenAI (fallback)", "gpt-4o-mini chat\ntext-embedding-3-small", "#f1f5f9", "#64748b", "#334155")
    draw_box(7.9, 0.5, 3.4, 1.4, "3. Degraded Mode (no LLM)", "Regex classifier, template answer,\nrule-based judge, BM25-only search", "#f1f5f9", "#64748b", "#334155")
    draw_box(11.8, 0.5, 3.7, 1.4, "Feedback Loop (SQLite)", "Ratings + helpful cases ->\nbounded case/crop re-rank boost", "#ecfdf5", "#10b981", "#065f46")
    ax.text(0.5, 2.15, "invoke_with_fallback: provider chain", fontsize=8.5, color="#475569", style="italic")

    # Phase banners (see docs/TWO_PHASE_FLOW.md)
    ax.text(0.5, 8.95, "PHASE 1: PREPARATION (once, make setup)", fontsize=9.5, fontweight="bold", color="#065f46")
    ax.text(15.5, 8.95, "saves cases, models, indexes to disk", fontsize=8.5, style="italic", color="#475569", ha="right")
    ax.plot([0.3, 15.7], [6.92, 6.92], linestyle="--", lw=1, color="#94a3b8", zorder=1)
    ax.text(0.5, 6.75, "PHASE 2: ANSWERING (every query, reads what Phase 1 saved)", fontsize=9.5, fontweight="bold", color="#1e40af")

    # Arrows connecting layers
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#64748b", edgecolor="#64748b", arrowstyle="->", lw=1.5),
            fontsize=8, ha="center", color="#475569", zorder=4
        )

    draw_arrow(3.7, 7.9, 4.2, 7.9)
    draw_arrow(7.4, 7.9, 7.9, 7.9)
    draw_arrow(11.3, 7.9, 11.8, 7.9)

    draw_arrow(3.7, 5.75, 4.2, 5.75)
    draw_arrow(7.4, 5.75, 7.9, 5.75)
    draw_arrow(11.3, 5.75, 11.8, 5.75)

    draw_arrow(13.65, 5.0, 2.1, 4.3)
    draw_arrow(3.7, 3.45, 4.2, 3.45)
    draw_arrow(7.4, 3.45, 7.9, 3.45)
    draw_arrow(11.3, 3.45, 11.8, 3.45)
    draw_arrow(3.7, 1.2, 4.2, 1.2)
    draw_arrow(7.4, 1.2, 7.9, 1.2)
    draw_arrow(13.65, 1.9, 2.1, 2.6, "feedback boost")

    # Save multiple formats
    pdf_path = os.path.join(output_dir, "architecture.pdf")
    png_path = os.path.join(output_dir, "architecture.png")
    jpg_path = os.path.join(output_dir, "architecture.jpg")

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=110)
    plt.savefig(jpg_path, bbox_inches="tight", dpi=110)
    plt.close()

    print(f"Architecture diagrams generated:\n- {pdf_path}\n- {png_path}\n- {jpg_path}")

if __name__ == "__main__":
    create_architecture_diagram(str(Path(__file__).resolve().parent.parent / "docs"))

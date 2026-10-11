import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Search, Filter, Sparkles, AlertCircle, RefreshCw, ChevronDown, ChevronUp, Bot, FileText } from "lucide-react";
import { PipelineQueryResponse, MetadataFilters, CaseDocument } from "../types/api";
import { api } from "../api/client";
import { RecommendationCard } from "../components/RecommendationCard";
import { CaseTable } from "../components/CaseTable";
import { ImpactCharts } from "../components/ImpactCharts";
import { JudgePanel } from "../components/JudgePanel";
import { AgentTraceTimeline } from "../components/AgentTraceTimeline";
import { FeedbackWidget } from "../components/FeedbackWidget";
import { CaseDetailsModal } from "../components/CaseDetailsModal";
import { YieldGapPanel } from "../components/YieldGapPanel";

interface Props {
  selectedProvider: string;
  selectedModel: string;
  selectedEmbedProvider: string;
  selectedEmbedModel: string;
  searchMode: "hybrid" | "semantic" | "keyword";
  topK: number;
  metadataOptions: any;
  onQueryComplete: (res: PipelineQueryResponse) => void;
}

const EXAMPLE_QUERIES = [
  "Which crops do well in Punjab during Kharif with N around 90, P 40, K 40, pH 6.5 and about 1200 mm rainfall?",
  "Best high-yield crops for Haryana in Rabi season with cool weather (18°C) and neutral soil pH 7.0?",
  "I have 650 mm rainfall and medium NPK in Karnataka. What should I grow in Kharif?",
  "Why does wheat yield differ between Punjab and Bihar in Rabi?",
  "Recommend alternative crops to rice in Punjab for Kharif season with lower water requirements."
];

export const AdvisorPage: React.FC<Props> = ({
  selectedProvider,
  selectedModel,
  selectedEmbedProvider,
  selectedEmbedModel,
  searchMode,
  topK,
  metadataOptions,
  onQueryComplete
}) => {
  const [query, setQuery] = useState(EXAMPLE_QUERIES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<MetadataFilters>({});
  const [result, setResult] = useState<PipelineQueryResponse | null>(null);
  const [selectedCaseModal, setSelectedCaseModal] = useState<CaseDocument | null>(null);

  const handleSearch = async (queryText: string = query) => {
    if (!queryText.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await api.queryPipeline({
        query: queryText,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
        top_k: topK,
        mode: searchMode,
        llm: { provider: selectedProvider, model: selectedModel },
        embedding: { provider: selectedEmbedProvider, model: selectedEmbedModel }
      });
      setResult(res);
      onQueryComplete(res);
    } catch (err: any) {
      setError(err.message || "Failed to execute advisory query.");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCaseId = (caseId: string) => {
    const found = result?.similar_cases.find((c) => c.case_id === caseId);
    if (found) {
      setSelectedCaseModal(found);
      return;
    }
    // Records cited by the yield-gap analysis are not always among the retrieved cases.
    const example = result?.yield_gap?.groups.flatMap((g) => g.example_cases || []).find((c) => c.case_id === caseId);
    if (example) {
      setSelectedCaseModal({ ...example, scores: { retrieval: 0, proximity: 0, yield_pct: 0, recency: 0, feedback: 0, final: 0 } });
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Search & Query Input Box */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm">
        <div className="flex items-center gap-2 text-forest-700 font-semibold text-xs uppercase tracking-wider mb-2">
          <Sparkles className="w-4 h-4 text-forest-600" />
          Natural Language Farming Advisory
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="space-y-4"
        >
          <div className="relative flex items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe your soil, region, rainfall, temperature, or farming scenario..."
              className="w-full bg-slate-50 border border-slate-200 rounded-2xl pl-12 pr-32 py-3.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-forest-500/20 focus:border-forest-500 transition"
            />
            <Search className="w-5 h-5 text-slate-400 absolute left-4" />

            <div className="absolute right-2 flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowFilters(!showFilters)}
                className={`p-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition ${
                  showFilters || Object.keys(filters).some((k) => (filters as any)[k])
                    ? "bg-forest-50 text-forest-800 border border-forest-200"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
                title="Explicit Metadata Filters"
              >
                <Filter className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Filters</span>
              </button>

              <button
                type="submit"
                disabled={loading}
                className="bg-forest-700 hover:bg-forest-800 text-white font-semibold px-5 py-2 rounded-xl text-sm shadow-sm transition flex items-center gap-2 disabled:opacity-50 cursor-pointer"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                <span>{loading ? "Analyzing..." : "Ask"}</span>
              </button>
            </div>
          </div>

          {/* Clickable Example Queries */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
            <span className="text-slate-400 font-medium mr-1">Try example:</span>
            {EXAMPLE_QUERIES.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setQuery(ex);
                  handleSearch(ex);
                }}
                className="bg-slate-100 hover:bg-forest-50 hover:text-forest-900 text-slate-600 px-2.5 py-1 rounded-lg transition-colors truncate max-w-xs border border-transparent hover:border-forest-200"
              >
                {ex}
              </button>
            ))}
          </div>

          {/* Collapsible Filter Accordion */}
          {showFilters && (
            <div className="pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs animate-in fade-in duration-150">
              <div>
                <label className="block text-slate-600 font-medium mb-1">Target Crop</label>
                <select
                  value={filters.crop || ""}
                  onChange={(e) => setFilters({ ...filters, crop: e.target.value || null })}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-800"
                >
                  <option value="">Any Crop</option>
                  {(metadataOptions?.crops || []).map((c: string) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-600 font-medium mb-1">Season</label>
                <select
                  value={filters.season || ""}
                  onChange={(e) => setFilters({ ...filters, season: e.target.value || null })}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-800"
                >
                  <option value="">Any Season</option>
                  {(metadataOptions?.seasons || []).map((s: string) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-600 font-medium mb-1">State / Region</label>
                <select
                  value={filters.state || ""}
                  onChange={(e) => setFilters({ ...filters, state: e.target.value || null })}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-800"
                >
                  <option value="">Any State</option>
                  {(metadataOptions?.states || []).map((st: string) => (
                    <option key={st} value={st}>
                      {st}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-600 font-medium mb-1">Soil pH Class</label>
                <select
                  value={filters.ph_class || ""}
                  onChange={(e) => setFilters({ ...filters, ph_class: e.target.value || null })}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-800 capitalize"
                >
                  <option value="">Any pH</option>
                  <option value="acidic">Acidic (&lt;6.0)</option>
                  <option value="neutral">Neutral (6.0-7.5)</option>
                  <option value="alkaline">Alkaline (&gt;7.5)</option>
                </select>
              </div>
            </div>
          )}
        </form>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-2xl flex items-start gap-3 text-xs animate-in fade-in">
          <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-bold">Validation / Execution Notice:</span>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Results Dashboard */}
      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Classification Banner */}
          <div className="bg-forest-900 text-white rounded-2xl p-4 shadow-sm flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-forest-300" />
              <span>
                Intent: <strong className="text-forest-200 uppercase">{result.classification.intent.replace(/_/g, " ")}</strong>
              </span>
              {result.classification.state && (
                <span>• State: <strong className="text-white">{result.classification.state}</strong></span>
              )}
              {result.classification.season && (
                <span>• Season: <strong className="text-white">{result.classification.season}</strong></span>
              )}
            </div>
            <div className="text-forest-300 text-[11px] flex items-center gap-3">
              {result.search_mode_used && (
                <span>
                  Search: <strong className="text-white">{result.search_mode_used}</strong>
                  {result.search_mode_used !== searchMode && " (fallback)"}
                </span>
              )}
              <span>Classifier Confidence: {(result.classification.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>

          {(result.warnings?.length ?? 0) > 0 && (
            <div className="bg-amber-50 border border-amber-200 text-amber-900 rounded-2xl p-3 text-xs space-y-1">
              {result.warnings!.map((w, i) => (
                <div key={i} className="flex items-start gap-2">
                  <AlertCircle className="w-3.5 h-3.5 text-amber-600 flex-shrink-0 mt-0.5" />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}

          {result.out_of_scope ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-6 text-sm text-slate-700">
              {result.explanation}
            </div>
          ) : (
          <>

          {/* Section 1: yield-gap answer for "why do yields differ" questions, else recommendation cards */}
          {result.yield_gap ? (
            <YieldGapPanel gap={result.yield_gap} onSelectCase={handleSelectCaseId} />
          ) : (
          <>

          <div>
            <h3 className="font-bold text-slate-900 text-base mb-3 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-forest-600" />
              Recommended Crops & Yield Forecasts
            </h3>
            {result.recommendations.length === 0 && (
              <div className="bg-white rounded-2xl border border-slate-200 p-5 text-xs text-slate-600">
                No historical cases matched these conditions. Try removing filters or widening the question.
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {result.recommendations.map((rec, idx) => (
                <RecommendationCard
                  key={rec.crop}
                  recommendation={rec}
                  rank={idx + 1}
                  onSelectCase={handleSelectCaseId}
                />
              ))}
            </div>
          </div>

          </>
          )}

          {/* Section 2: Grounded Advisory Explanation */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
            <h4 className="font-bold text-slate-900 text-sm mb-3 flex items-center gap-2">
              <FileText className="w-4 h-4 text-forest-600" />
              Agronomic Advisory Synthesis & Case Evidence
            </h4>
            <div className="prose prose-slate prose-sm max-w-none prose-headings:text-slate-900 prose-headings:font-semibold prose-h2:text-base prose-h3:text-sm prose-p:text-slate-700 prose-p:leading-relaxed prose-li:text-slate-700 prose-strong:text-slate-900 prose-table:text-xs prose-th:bg-slate-50 prose-th:px-3 prose-th:py-2 prose-td:px-3 prose-td:py-2 prose-code:bg-slate-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-forest-800 prose-code:font-medium prose-code:before:content-none prose-code:after:content-none prose-a:text-forest-700 prose-a:no-underline hover:prose-a:underline prose-blockquote:border-forest-300 prose-blockquote:text-slate-600 prose-hr:border-slate-200">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.explanation}
              </ReactMarkdown>
            </div>
          </div>

          {/* Section 3: Impact Analysis & Sensitivity Charts */}
          {result.impact_analysis && (
            <ImpactCharts impact={result.impact_analysis} />
          )}

          {/* Section 4: Judge Validation & Agent Trace */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <JudgePanel judge={result.judge} />
            <AgentTraceTimeline trace={result.agent_trace} />
          </div>

          {/* Section 5: Similar Cases Table */}
          <CaseTable
            cases={result.similar_cases}
            onSelectCase={handleSelectCaseId}
          />

          {/* Section 6: Feedback Loop Widget */}
          <FeedbackWidget
            query={query}
            cases={result.similar_cases}
            topCrop={result.recommendations[0]?.crop}
          />

          </>
          )}

          {/* Section 7: Disclaimer */}
          <div className="p-4 bg-slate-100 rounded-xl text-[11px] text-slate-500 leading-relaxed border border-slate-200">
            {result.disclaimer}
          </div>
        </div>
      )}

      {/* Case Details Popup Modal */}
      <CaseDetailsModal
        caseDoc={selectedCaseModal}
        onClose={() => setSelectedCaseModal(null)}
      />
    </div>
  );
};

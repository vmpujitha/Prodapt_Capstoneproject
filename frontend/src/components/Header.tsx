import React, { useState } from "react";
import { Sprout, Settings as SettingsIcon, Cpu, Layers, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";
import { ProviderStatus, IndexJob, IndexStatus } from "../types/api";

interface HeaderProps {
  providers: ProviderStatus[];
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (provider: string, model: string) => void;
  selectedEmbedProvider: string;
  selectedEmbedModel: string;
  onEmbedChange: (provider: string, model: string) => void;
  searchMode: "hybrid" | "semantic" | "keyword";
  onSearchModeChange: (mode: "hybrid" | "semantic" | "keyword") => void;
  topK: number;
  onTopKChange: (k: number) => void;
  onRebuildIndex: () => Promise<void>;
  isReindexing: boolean;
  indexStatus?: IndexStatus | null;
  indexJob?: IndexJob | null;
  indexError?: string | null;
  lastLlmUsed?: { provider?: string; model?: string } | null;
  degradedMode?: boolean;
  fallbackTriggered?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  providers,
  selectedProvider,
  selectedModel,
  onProviderChange,
  selectedEmbedProvider,
  selectedEmbedModel,
  onEmbedChange,
  searchMode,
  onSearchModeChange,
  topK,
  onTopKChange,
  onRebuildIndex,
  isReindexing,
  indexStatus,
  indexJob,
  indexError,
  lastLlmUsed,
  degradedMode,
  fallbackTriggered
}) => {
  const [showSettings, setShowSettings] = useState(false);

  const activeProv = providers.find((p) => p.provider === selectedProvider);
  const models = activeProv?.chat_models || [];
  const indexMissing = indexStatus ? !indexStatus.up_to_date : false;
  const embeddingsDown = indexStatus ? !indexStatus.embeddings_available : false;

  return (
    <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-forest-600 to-forest-800 flex items-center justify-center text-white shadow-md shadow-forest-900/10">
            <Sprout className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 leading-tight tracking-tight flex items-center gap-2">
              AgriAdvisor
              <span className="text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 bg-forest-100 text-forest-800 rounded-full border border-forest-200">
                Multi-Agent AI
              </span>
            </h1>
            <p className="text-xs text-slate-500 font-medium">Agricultural Decision Support & Crop Yield System</p>
          </div>
        </div>

        {/* Center / Right controls */}
        <div className="flex items-center gap-3">
          {/* Runtime Execution Status Badge */}
          {lastLlmUsed !== undefined && (
            <div className="hidden md:flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border bg-slate-50 border-slate-200">
              {degradedMode ? (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                  <span className="text-amber-800 font-medium">Degraded Mode (Deterministic)</span>
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-forest-600" />
                  <span className="text-slate-700">
                    <span className="font-semibold text-forest-700">{lastLlmUsed?.provider}</span> ({lastLlmUsed?.model})
                  </span>
                  {fallbackTriggered && (
                    <span className="ml-1 text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.2 rounded font-semibold">
                      Fallback
                    </span>
                  )}
                </>
              )}
            </div>
          )}

          {/* Provider Dropdown */}
          <div className="flex items-center gap-1.5 bg-slate-100/80 p-1 rounded-xl border border-slate-200 text-xs">
            <div className="flex items-center gap-1 px-2 text-slate-500 font-medium">
              <Cpu className="w-3.5 h-3.5 text-slate-600" />
              <span>LLM:</span>
            </div>
            <select
              value={selectedProvider}
              onChange={(e) => {
                const prov = providers.find((p) => p.provider === e.target.value);
                const defaultMdl = prov?.default_chat_model || prov?.chat_models[0] || "";
                onProviderChange(e.target.value, defaultMdl);
              }}
              className="bg-white text-slate-800 font-semibold px-2 py-1 rounded-lg border border-slate-200 focus:outline-none focus:ring-1 focus:ring-forest-500 cursor-pointer"
            >
              {providers.map((p) => (
                <option key={p.provider} value={p.provider} disabled={!p.available} title={p.reason || undefined}>
                  {p.provider === "ollama" ? "Ollama (Local)" : "OpenAI (Cloud)"} {!p.available ? `(${p.reason || "Unavailable"})` : ""}
                </option>
              ))}
            </select>

            {/* Model Selection */}
            {models.length > 0 && (
              <select
                value={selectedModel}
                onChange={(e) => onProviderChange(selectedProvider, e.target.value)}
                className="bg-white text-slate-800 font-medium px-2 py-1 rounded-lg border border-slate-200 focus:outline-none focus:ring-1 focus:ring-forest-500 cursor-pointer"
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Advanced Settings Popover Button */}
          <div className="relative">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-xl border transition-all ${
                showSettings
                  ? "bg-forest-50 border-forest-300 text-forest-700 shadow-inner"
                  : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
              title="Advanced Settings"
            >
              <SettingsIcon className="w-4 h-4" />
            </button>

            {showSettings && (
              <div className="absolute right-0 mt-2 w-80 bg-white rounded-2xl shadow-xl border border-slate-200 p-4 z-50 text-xs animate-in fade-in zoom-in-95 duration-150">
                <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-100">
                  <h3 className="font-semibold text-slate-900 text-sm flex items-center gap-1.5">
                    <Layers className="w-4 h-4 text-forest-600" />
                    Retrieval & Embedding Settings
                  </h3>
                </div>

                <div className="space-y-3">
                  {/* Search Mode */}
                  <div>
                    <label className="block text-slate-600 font-medium mb-1">Search Retrieval Mode</label>
                    <div className="grid grid-cols-3 gap-1 bg-slate-100 p-1 rounded-lg">
                      {(["hybrid", "semantic", "keyword"] as const).map((m) => (
                        <button
                          key={m}
                          onClick={() => onSearchModeChange(m)}
                          className={`py-1 rounded text-center capitalize font-medium transition-all ${
                            searchMode === m
                              ? "bg-white text-forest-800 shadow-sm font-semibold"
                              : "text-slate-600 hover:text-slate-900"
                          }`}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Embedding Provider */}
                  <div>
                    <label className="block text-slate-600 font-medium mb-1">Embedding Provider</label>
                    <select
                      value={selectedEmbedProvider}
                      onChange={(e) => {
                        const prov = providers.find((p) => p.provider === e.target.value);
                        onEmbedChange(e.target.value, prov?.default_embedding_model || prov?.embedding_models[0] || "");
                      }}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-800"
                    >
                      {(providers.length ? providers : [{ provider: "ollama" }, { provider: "openai" }] as ProviderStatus[]).map((p) => (
                        <option
                          key={p.provider}
                          value={p.provider}
                          disabled={p.available === false}
                          title={p.reason || undefined}
                        >
                          {p.provider === "ollama" ? "Ollama" : "OpenAI"} ({p.default_embedding_model || p.embedding_models?.[0] || "default"})
                          {p.available === false ? " - unavailable" : ""}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Top-K */}
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-slate-600 font-medium">Top-K Cases Retrieved</label>
                      <span className="font-semibold text-forest-700">{topK}</span>
                    </div>
                    <input
                      type="range"
                      min="3"
                      max="20"
                      value={topK}
                      onChange={(e) => onTopKChange(parseInt(e.target.value))}
                      className="w-full accent-forest-600 cursor-pointer"
                    />
                  </div>

                  {/* Vector index status / build */}
                  <div className="pt-2 border-t border-slate-100 space-y-2">
                    {indexStatus && (
                      <div className="text-[11px] text-slate-600">
                        <div className="font-mono text-slate-500 truncate" title={indexStatus.collection}>{indexStatus.collection}</div>
                        {indexStatus.up_to_date ? (
                          <span className="text-forest-700 font-medium">Ready: {indexStatus.document_count.toLocaleString()} cases indexed</span>
                        ) : (
                          <span className="text-amber-700 font-medium">
                            {indexStatus.exists
                              ? `Stale: ${indexStatus.document_count.toLocaleString()} of ${indexStatus.expected_documents.toLocaleString()} cases`
                              : "Not built: semantic/hybrid search falls back to keyword search"}
                          </span>
                        )}
                        {embeddingsDown && <div className="text-rose-600 mt-0.5">{indexStatus.reason}</div>}
                      </div>
                    )}
                    {isReindexing && indexJob && (
                      <div>
                        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div className="h-full bg-forest-600 transition-all" style={{ width: `${Math.round(indexJob.progress * 100)}%` }} />
                        </div>
                        <div className="text-[10px] text-slate-500 mt-0.5">
                          {indexJob.indexed.toLocaleString()} / {indexJob.total.toLocaleString()} documents
                        </div>
                      </div>
                    )}
                    {indexError && <div className="text-[11px] text-rose-600">{indexError}</div>}
                    <button
                      onClick={onRebuildIndex}
                      disabled={isReindexing || embeddingsDown}
                      title={embeddingsDown ? indexStatus?.reason || undefined : undefined}
                      className="w-full flex items-center justify-center gap-2 bg-forest-600 hover:bg-forest-700 text-white font-medium py-1.5 px-3 rounded-lg shadow-sm transition disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${isReindexing ? "animate-spin" : ""}`} />
                      {isReindexing ? "Building index..." : indexMissing ? "Build index" : "Rebuild index"}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

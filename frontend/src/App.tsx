import React, { useState, useEffect, useCallback } from "react";
import { Sprout, Scale, Award, MessageSquare } from "lucide-react";
import { Header } from "./components/Header";
import { SyntheticBanner } from "./components/SyntheticBanner";
import { AdvisorPage } from "./pages/AdvisorPage";
import { ComparePage } from "./pages/ComparePage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { FeedbackPage } from "./pages/FeedbackPage";
import { ProviderStatus, PipelineQueryResponse, IndexJob, IndexStatus } from "./types/api";
import { api } from "./api/client";

type TabType = "advisor" | "compare" | "eval" | "feedback";

export function App() {
  const [tab, setTab] = useState<TabType>("advisor");
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>(
    localStorage.getItem("agri_llm_provider") || "ollama"
  );
  const [selectedModel, setSelectedModel] = useState<string>(
    localStorage.getItem("agri_llm_model") || "llama3.1:8b"
  );
  const [selectedEmbedProvider, setSelectedEmbedProvider] = useState<string>(
    localStorage.getItem("agri_embed_provider") || "ollama"
  );
  const [selectedEmbedModel, setSelectedEmbedModel] = useState<string>(
    localStorage.getItem("agri_embed_model") || "nomic-embed-text"
  );
  const [searchMode, setSearchMode] = useState<"hybrid" | "semantic" | "keyword">("hybrid");
  const [topK, setTopK] = useState<number>(8);
  const [isSynthetic, setIsSynthetic] = useState<boolean>(false);
  const [metadataOptions, setMetadataOptions] = useState<any>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexJob, setIndexJob] = useState<IndexJob | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);

  // Runtime response tracking
  const [lastQueryRes, setLastQueryRes] = useState<PipelineQueryResponse | null>(null);

  useEffect(() => {
    // 1. Fetch Health to detect synthetic dataset status
    api.getHealth()
      .then((h) => {
        setIsSynthetic(h.is_synthetic_data);
      })
      .catch((err) => console.error("Health check error:", err));

    // 2. Fetch Provider Statuses
    api.getProviders()
      .then((provs) => {
        setProviders(provs);
        const storedProv = localStorage.getItem("agri_llm_provider");
        const found = provs.find((p) => p.provider === storedProv);
        if (found && found.available) {
          setSelectedProvider(found.provider);
          setSelectedModel(found.default_chat_model || found.chat_models[0] || "");
        } else {
          // Default to first available or ollama
          const avail = provs.find((p) => p.available) || provs[0];
          if (avail) {
            setSelectedProvider(avail.provider);
            setSelectedModel(avail.default_chat_model || avail.chat_models[0] || "");
          }
        }
      })
      .catch((err) => console.error("Providers fetch error:", err));

    // 3. Fetch Filter Metadata Options
    api.getMetadataOptions()
      .then((opts) => setMetadataOptions(opts))
      .catch((err) => console.error("Metadata options error:", err));
  }, []);

  const handleProviderChange = (prov: string, model: string) => {
    setSelectedProvider(prov);
    setSelectedModel(model);
    localStorage.setItem("agri_llm_provider", prov);
    localStorage.setItem("agri_llm_model", model);
  };

  const handleEmbedChange = (prov: string, model: string) => {
    setSelectedEmbedProvider(prov);
    setSelectedEmbedModel(model);
    localStorage.setItem("agri_embed_provider", prov);
    localStorage.setItem("agri_embed_model", model);
  };

  const refreshIndexStatus = useCallback(() => {
    api.getIndexStatus(selectedEmbedProvider, selectedEmbedModel)
      .then((status) => {
        setIndexStatus(status);
        if (status.job) setIndexJob(status.job);
      })
      .catch(() => setIndexStatus(null));
  }, [selectedEmbedProvider, selectedEmbedModel]);

  useEffect(() => {
    refreshIndexStatus();
  }, [refreshIndexStatus]);

  // Poll the background index build while it runs.
  useEffect(() => {
    if (!indexJob || indexJob.status !== "running") return;
    const timer = setInterval(async () => {
      try {
        const job = await api.getReindexStatus(indexJob.job_id);
        setIndexJob(job);
        if (job.status !== "running") {
          if (job.status === "failed") setIndexError(job.error || "Index build failed");
          refreshIndexStatus();
        }
      } catch (err: any) {
        setIndexError(err.message);
        setIndexJob(null);
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [indexJob, refreshIndexStatus]);

  const handleRebuildIndex = async () => {
    setIndexError(null);
    try {
      setIndexJob(await api.reindex(selectedEmbedProvider, selectedEmbedModel));
    } catch (err: any) {
      setIndexError(err.message);
    }
  };

  const tabs: { id: TabType; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: "advisor", label: "Advisor", icon: Sprout },
    { id: "compare", label: "Compare Farms", icon: Scale },
    { id: "eval", label: "Evaluation", icon: Award },
    { id: "feedback", label: "Feedback Loop", icon: MessageSquare }
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Synthetic Banner */}
      <SyntheticBanner isSynthetic={isSynthetic} />

      {/* Global Header with Provider Switcher & Settings */}
      <Header
        providers={providers}
        selectedProvider={selectedProvider}
        selectedModel={selectedModel}
        onProviderChange={handleProviderChange}
        selectedEmbedProvider={selectedEmbedProvider}
        selectedEmbedModel={selectedEmbedModel}
        onEmbedChange={handleEmbedChange}
        searchMode={searchMode}
        onSearchModeChange={setSearchMode}
        topK={topK}
        onTopKChange={setTopK}
        onRebuildIndex={handleRebuildIndex}
        isReindexing={indexJob?.status === "running"}
        indexStatus={indexStatus}
        indexJob={indexJob}
        indexError={indexError}
        lastLlmUsed={lastQueryRes ? lastQueryRes.llm_used : undefined}
        degradedMode={lastQueryRes?.degraded_mode}
        fallbackTriggered={lastQueryRes?.fallback_triggered}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-200/60 rounded-2xl w-fit mb-6 text-xs">
          {tabs.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-semibold transition-all cursor-pointer ${
                  active
                    ? "bg-white text-forest-800 shadow-xs shadow-slate-900/5"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/40"
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? "text-forest-600" : "text-slate-500"}`} />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Views */}
        {tab === "advisor" && (
          <AdvisorPage
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
            selectedEmbedProvider={selectedEmbedProvider}
            selectedEmbedModel={selectedEmbedModel}
            searchMode={searchMode}
            topK={topK}
            metadataOptions={metadataOptions}
            onQueryComplete={(res) => setLastQueryRes(res)}
          />
        )}

        {tab === "compare" && <ComparePage metadataOptions={metadataOptions} />}

        {tab === "eval" && (
          <EvaluationPage
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
          />
        )}

        {tab === "feedback" && <FeedbackPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <p>
          AI-Powered Agriculture Crop Advisory Assistant • LangChain & LangGraph Multi-Agent Architecture
        </p>
      </footer>
    </div>
  );
}

export default App;

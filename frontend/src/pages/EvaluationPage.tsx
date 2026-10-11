import React, { useState, useEffect } from "react";
import { Play, CheckCircle2, XCircle, Award, Target, RefreshCw, BarChart2, ShieldCheck, AlertTriangle } from "lucide-react";
import { api, ApiError } from "../api/client";
import { EvalStatusResponse } from "../types/api";

interface Props {
  selectedProvider: string;
  selectedModel: string;
}

// Missing metrics are shown as "not run" - never as a placeholder number.
const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "not run" : `${(v * 100).toFixed(digits)}%`;
const score = (v: number | null | undefined) => (v === null || v === undefined ? "–" : v.toFixed(2));

const MetricCard: React.FC<{ icon: React.ReactNode; label: string; value: string; hint: string }> = ({ icon, label, value, hint }) => (
  <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-2xs">
    <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 mb-1">
      {icon}
      {label}
    </div>
    <div className={`text-2xl font-extrabold ${value === "not run" ? "text-slate-400 text-lg" : "text-slate-900"}`}>{value}</div>
    <span className="text-[11px] text-slate-400">{hint}</span>
  </div>
);

export const EvaluationPage: React.FC<Props> = ({ selectedProvider, selectedModel }) => {
  const [evalData, setEvalData] = useState<EvalStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [noRunsYet, setNoRunsYet] = useState(false);

  useEffect(() => {
    api.getLatestEval()
      .then((res) => setEvalData(res))
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) setNoRunsYet(true);
        else setError(err.message);
      });
  }, []);

  useEffect(() => {
    if (!runningJobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await api.getEvalStatus(runningJobId);
        setEvalData(status);
        if (status.status === "completed" || status.status === "failed") {
          setRunningJobId(null);
          setLoading(false);
          if (status.status === "failed") setError(status.error || "Evaluation failed");
        }
      } catch (err: any) {
        setError(err.message);
        setRunningJobId(null);
        setLoading(false);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [runningJobId]);

  const handleRunEvaluation = async () => {
    setLoading(true);
    setError(null);
    setNoRunsYet(false);
    try {
      const res = await api.triggerEval(undefined, { provider: selectedProvider, model: selectedModel });
      setRunningJobId(res.run_id);
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  const agg = evalData?.aggregate_metrics || {};
  const deepevalRan = evalData?.deepeval_status === "ran";

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-forest-700 font-semibold text-xs uppercase tracking-wider mb-1">
            <Award className="w-4 h-4 text-forest-600" />
            DeepEval Benchmark Suite
          </div>
          <h2 className="text-xl font-bold text-slate-900">Golden Set Evaluation</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Runs every golden query through the full agent pipeline. Retrieval hit-rate is always computed; DeepEval
            metrics (answer relevancy, faithfulness, contextual relevancy/precision, G-Eval) need a reachable judge LLM.
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={loading}
          className="bg-forest-700 hover:bg-forest-800 text-white font-semibold py-2.5 px-6 rounded-2xl shadow-sm transition flex items-center gap-2 cursor-pointer disabled:opacity-50 text-xs"
        >
          {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
          <span>{loading ? "Running evaluation..." : "Run evaluation"}</span>
        </button>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-2xl p-4 text-xs">{error}</div>
      )}

      {noRunsYet && !evalData && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 text-xs text-slate-600">
          No evaluation has been run yet. Click <strong>Run evaluation</strong> to benchmark the pipeline. With a local
          8B model this can take several minutes per query.
        </div>
      )}

      {(loading || evalData?.status === "running") && (
        <div className="bg-white rounded-2xl border border-forest-200 p-4 shadow-xs text-xs">
          <div className="flex justify-between items-center mb-1 font-semibold text-forest-800">
            <span>
              Evaluating {evalData?.completed_samples ?? 0} / {evalData?.total_samples ?? "?"} queries in the background...
            </span>
            <span>{Math.round((evalData?.progress || 0) * 100)}%</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className="bg-forest-600 h-full transition-all duration-300" style={{ width: `${Math.max(3, (evalData?.progress || 0) * 100)}%` }} />
          </div>
        </div>
      )}

      {evalData && evalData.status === "completed" && (
        <>
          <div className="flex flex-wrap gap-2 text-[11px]">
            <span className={`px-2 py-1 rounded-lg border ${deepevalRan ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-amber-50 border-amber-200 text-amber-800"}`}>
              DeepEval: {deepevalRan ? `ran with ${evalData.judge_model}` : evalData.deepeval_status || "unknown"}
            </span>
            <span className="px-2 py-1 rounded-lg border bg-slate-50 border-slate-200 text-slate-600">
              Run <code className="font-mono">{evalData.run_id}</code> · {evalData.completed_at?.slice(0, 19).replace("T", " ")} UTC
            </span>
            {evalData.is_synthetic_data && (
              <span className="px-2 py-1 rounded-lg border bg-amber-50 border-amber-300 text-amber-900 font-semibold">
                <AlertTriangle className="w-3 h-3 inline mr-1" />Evaluated on SYNTHETIC data
              </span>
            )}
            {!!agg.degraded_runs && (
              <span className="px-2 py-1 rounded-lg border bg-amber-50 border-amber-200 text-amber-800">
                {agg.degraded_runs} of {agg.total_evaluated} pipeline runs used degraded (no-LLM) mode
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard icon={<Target className="w-4 h-4 text-forest-600" />} label={`Hit-Rate @ ${agg.k ?? 5}`}
              value={pct(agg.hit_rate_at_k)} hint="An expected crop appears among the top retrieved cases" />
            <MetricCard icon={<Target className="w-4 h-4 text-forest-600" />} label="Recommendation Hit-Rate"
              value={pct(agg.recommendation_hit_rate)} hint="An expected crop appears among the recommendations" />
            <MetricCard icon={<ShieldCheck className="w-4 h-4 text-indigo-600" />} label="Judge Pass Rate"
              value={pct(agg.judge_pass_rate)} hint="Recommendations accepted by the in-pipeline judge" />
            <MetricCard icon={<BarChart2 className="w-4 h-4 text-emerald-600" />} label="Answer Relevancy (DeepEval)"
              value={pct(agg.answer_relevancy, 1)} hint="Answer addresses the question" />
            <MetricCard icon={<ShieldCheck className="w-4 h-4 text-indigo-600" />} label="Faithfulness (DeepEval)"
              value={pct(agg.faithfulness, 1)} hint="Claims supported by the retrieved cases" />
            <MetricCard icon={<BarChart2 className="w-4 h-4 text-emerald-600" />} label="Contextual Relevancy (DeepEval)"
              value={pct(agg.contextual_relevancy, 1)} hint="Retrieved cases are relevant to the question" />
            <MetricCard icon={<BarChart2 className="w-4 h-4 text-emerald-600" />} label="Contextual Precision (DeepEval)"
              value={pct(agg.contextual_precision, 1)} hint="Relevant cases are ranked first" />
            <MetricCard icon={<Award className="w-4 h-4 text-amber-500" />} label="CropRecommendationQuality (G-Eval)"
              value={pct(agg.crop_recommendation_quality_geval, 1)} hint="Custom G-Eval agronomic quality criteria" />
          </div>
        </>
      )}

      {evalData?.results && evalData.results.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
          <div className="p-4 border-b border-slate-100">
            <h3 className="font-bold text-slate-900 text-sm">Per-query results ({evalData.results.length})</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                  <th className="py-3 px-4">Hit</th>
                  <th className="py-3 px-4">Scenario</th>
                  <th className="py-3 px-4">Query</th>
                  <th className="py-3 px-4">Expected</th>
                  <th className="py-3 px-4">Recommended</th>
                  <th className="py-3 px-4 text-right">Relevancy</th>
                  <th className="py-3 px-4 text-right">Faithfulness</th>
                  <th className="py-3 px-4 text-right">G-Eval</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {evalData.results.map((r, i) => (
                  <tr key={r.id || i} className="hover:bg-slate-50/60 transition align-top">
                    <td className="py-3 px-4" title={r.error || r.reason || undefined}>
                      {r.error ? (
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                      ) : r.hit_at_k ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      ) : (
                        <XCircle className="w-4 h-4 text-rose-600" />
                      )}
                    </td>
                    <td className="py-3 px-4 capitalize text-slate-600">{r.scenario.replace(/_/g, " ")}</td>
                    <td className="py-3 px-4 max-w-sm font-medium text-slate-900">{r.query}</td>
                    <td className="py-3 px-4">
                      <div className="flex gap-1 flex-wrap">
                        {(r.expected_crops_resolved?.length ? r.expected_crops_resolved : r.expected_crops).map((c) => (
                          <span key={c} className="px-1.5 py-0.5 bg-slate-100 text-slate-700 rounded text-[11px]">{c}</span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-slate-600">{(r.recommended_crops || []).slice(0, 4).join(", ") || "–"}</td>
                    <td className="py-3 px-4 text-right">{score(r.relevancy_score)}</td>
                    <td className="py-3 px-4 text-right">{score(r.faithfulness_score)}</td>
                    <td className="py-3 px-4 text-right">{score(r.geval_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

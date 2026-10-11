import React, { useState } from "react";
import { TrendingUp, ShieldAlert, ShieldCheck, ChevronDown, ChevronUp, Tag, Sparkles } from "lucide-react";
import { CropRecommendation } from "../types/api";
import { yieldUnit } from "../utils/units";

interface Props {
  recommendation: CropRecommendation;
  rank: number;
  onSelectCase: (caseId: string) => void;
}

export const RecommendationCard: React.FC<Props> = ({ recommendation, rank, onSelectCase }) => {
  const [expanded, setExpanded] = useState(false);
  const r = recommendation;
  const unit = r.predicted_yield?.unit || yieldUnit(r.crop);

  const scorePct = Math.round(r.score * 100);

  const getRiskBadge = (level?: string) => {
    switch (level) {
      case "low":
        return {
          bg: "bg-emerald-50 text-emerald-700 border-emerald-200",
          icon: ShieldCheck,
          text: "Low Climate/Soil Risk"
        };
      case "medium":
        return {
          bg: "bg-amber-50 text-amber-700 border-amber-200",
          icon: ShieldAlert,
          text: "Moderate Climate Risk"
        };
      case "high":
        return {
          bg: "bg-rose-50 text-rose-700 border-rose-200",
          icon: ShieldAlert,
          text: "Elevated Risk"
        };
      default:
        return {
          bg: "bg-slate-50 text-slate-700 border-slate-200",
          icon: ShieldCheck,
          text: "Normal"
        };
    }
  };

  const riskBadge = getRiskBadge(r.risk?.level);
  const RiskIcon = riskBadge.icon;

  return (
    <div
      className={`rounded-2xl border transition-all duration-200 bg-white overflow-hidden ${
        rank === 1
          ? "border-forest-400 shadow-md shadow-forest-900/5 ring-1 ring-forest-400/30"
          : "border-slate-200 shadow-sm hover:border-slate-300"
      }`}
    >
      {/* Header */}
      <div className="p-5 pb-4 flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm ${
              rank === 1
                ? "bg-forest-600 text-white shadow-sm shadow-forest-800/20"
                : "bg-slate-100 text-slate-700 font-semibold"
            }`}
          >
            #{rank}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xl font-bold text-slate-900 tracking-tight">{r.crop}</h3>
              {rank === 1 && (
                <span className="flex items-center gap-1 text-[11px] font-semibold text-forest-700 bg-forest-50 px-2 py-0.5 rounded-full border border-forest-200">
                  <Sparkles className="w-3 h-3 text-forest-600" />
                  Top Match
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              Based on {r.case_count} matching historical record(s)
            </p>
          </div>
        </div>

        {/* Score Pill */}
        <div className="text-right">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-forest-50 border border-forest-200 rounded-xl">
            <span className="text-xs font-medium text-forest-700">Suitability</span>
            <span className="text-base font-extrabold text-forest-800">{scorePct}%</span>
          </div>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="px-5 py-3 grid grid-cols-2 gap-3 bg-slate-50/50 border-y border-slate-100">
        {/* ML Predicted Yield */}
        <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 font-medium mb-1">
            <TrendingUp className="w-3.5 h-3.5 text-forest-600" />
            <span>ML Predicted Yield</span>
          </div>
          {r.predicted_yield ? (
            <div>
              <div className="text-base font-bold text-slate-900">
                {r.predicted_yield.value.toFixed(2)} <span className="text-xs font-normal text-slate-500">{unit}</span>
              </div>
              <div className="text-[11px] text-slate-400 font-medium">
                Interval: {r.predicted_yield.low.toFixed(1)} – {r.predicted_yield.high.toFixed(1)} {unit}
              </div>
            </div>
          ) : (
            <div className="text-base font-bold text-slate-900">
              {r.median_yield.toFixed(2)} <span className="text-xs font-normal text-slate-500">{unit} (median)</span>
            </div>
          )}
        </div>

        {/* Risk Assessment */}
        <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="text-xs text-slate-500 font-medium mb-1 flex items-center justify-between">
            <span>Crop Risk Status</span>
            {r.risk && (
              <span className="text-[11px] text-slate-400">P(low): {Math.round(r.risk.probability * 100)}%</span>
            )}
          </div>
          <div className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-lg border ${riskBadge.bg}`}>
            <RiskIcon className="w-3.5 h-3.5" />
            <span>{riskBadge.text}</span>
          </div>
        </div>
      </div>

      {/* Supporting Case Tags */}
      <div className="p-5 pt-3">
        <div className="flex items-center justify-between">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-slate-400 font-medium flex items-center gap-1 mr-1">
              <Tag className="w-3 h-3" />
              Evidence:
            </span>
            {r.supporting_cases.map((cid) => (
              <button
                key={cid}
                onClick={() => onSelectCase(cid)}
                className="text-xs font-mono font-semibold px-2 py-0.5 bg-slate-100 text-forest-800 hover:bg-forest-100 hover:text-forest-900 rounded-md transition-colors border border-slate-200 hover:border-forest-300"
              >
                {cid}
              </button>
            ))}
          </div>

          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs font-medium text-slate-500 hover:text-slate-800 flex items-center gap-0.5"
          >
            <span>{expanded ? "Less" : "Details"}</span>
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Expanded Analysis Breakdown */}
        {expanded && (
          <div className="mt-4 pt-3 border-t border-slate-100 text-xs space-y-2.5 animate-in fade-in duration-150">
            {/* Risk factors list */}
            {r.risk?.factors && r.risk.factors.length > 0 && (
              <div className="space-y-1">
                <span className="font-semibold text-slate-700">Identified Risk Factors:</span>
                <ul className="list-disc list-inside space-y-0.5 text-slate-600 pl-1">
                  {r.risk.factors.map((f, i) => (
                    <li key={i}>{f.description}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Score breakdown stats */}
            {r.score_breakdown && (
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] bg-slate-50 p-2 rounded-lg border border-slate-100">
                <div>
                  <span className="text-slate-400 block">Retrieval Sim</span>
                  <span className="font-semibold text-slate-800">{r.similarity.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Hist. Percentile</span>
                  <span className="font-semibold text-slate-800">{r.yield_percentile.toFixed(0)}th %</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Hist. Mean Yield</span>
                  <span className="font-semibold text-slate-800">{r.mean_yield.toFixed(2)} {unit}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { Scale, ArrowRight, Info } from "lucide-react";
import { YieldGapAnalysis } from "../types/api";

interface Props {
  gap: YieldGapAnalysis;
  onSelectCase: (caseId: string) => void;
}

const CONDITION_LABELS: Record<string, string> = {
  rainfall: "Rainfall (mm)",
  temperature: "Temperature (°C)",
  humidity: "Humidity (%)",
  n: "N (kg/ha)",
  p: "P (kg/ha)",
  k: "K (kg/ha)",
  ph: "Soil pH",
  fertilizer: "Fertilizer (kg/ha)",
  pesticide: "Pesticide (kg/ha)",
};

export const YieldGapPanel: React.FC<Props> = ({ gap, onSelectCase }) => {
  const unit = gap.yield_unit || "t/ha";
  const chartData = gap.ranked_factors
    .filter((f) => f.estimated_impact_on_yield !== 0)
    .map((f) => ({ factor: f.factor.split("(")[0].trim(), impact: f.estimated_impact_on_yield }));
  const maxYield = Math.max(...gap.groups.map((g) => g.median_yield), 0.01);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
            <Scale className="w-4 h-4 text-forest-600" />
            Yield Gap Analysis: {gap.crop}
            {gap.season ? ` (${gap.season})` : ""}
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {gap.grouping === "state"
              ? "Median of all historical records per state"
              : "Best vs worst historical seasons for this crop"}
          </p>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-500 block">Yield gap</span>
          <span className={`text-2xl font-extrabold ${gap.yield_gap >= 0 ? "text-forest-700" : "text-rose-600"}`}>
            {gap.yield_gap > 0 ? "+" : ""}
            {gap.yield_gap.toFixed(2)} {unit}
          </span>
        </div>
      </div>

      {/* Group comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {gap.groups.map((g) => (
          <div key={g.name} className="rounded-xl border border-slate-200 p-4 bg-slate-50/60">
            <div className="flex items-baseline justify-between">
              <span className="font-semibold text-slate-900 text-sm">{g.name}</span>
              <span className="text-xs text-slate-500">{g.case_count} records</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-forest-600" style={{ width: `${(g.median_yield / maxYield) * 100}%` }} />
              </div>
              <span className="text-sm font-bold text-forest-800 whitespace-nowrap">
                {g.median_yield.toFixed(2)} {unit}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-x-3 gap-y-1 text-[11px] text-slate-600">
              {Object.entries(CONDITION_LABELS).map(([key, label]) =>
                g.conditions[key] !== undefined ? (
                  <div key={key} className="flex justify-between gap-1">
                    <span className="text-slate-400 truncate">{label}</span>
                    <span className="font-medium">{Number(g.conditions[key]).toLocaleString()}</span>
                  </div>
                ) : null
              )}
            </div>
            <div className="mt-3 flex flex-wrap gap-1">
              {g.example_case_ids.map((cid) => (
                <button
                  key={cid}
                  onClick={() => onSelectCase(cid)}
                  className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-white border border-slate-200 text-slate-600 hover:border-forest-400"
                >
                  {cid}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Factor contributions */}
      {chartData.length > 0 ? (
        <div>
          <h4 className="font-semibold text-slate-900 text-xs mb-2">
            What accounts for the gap (model estimate of each factor on its own)
          </h4>
          <div className="w-full" style={{ height: Math.max(140, chartData.length * 34) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30, top: 5, bottom: 5 }}>
                <XAxis type="number" unit={` ${unit}`} tick={{ fontSize: 10, fill: "#64748b" }} />
                <YAxis type="category" dataKey="factor" tick={{ fontSize: 10, fill: "#334155" }} width={110} />
                <ReferenceLine x={0} stroke="#94a3b8" />
                <Tooltip
                  formatter={(val: any) => [`${Number(val) > 0 ? "+" : ""}${Number(val).toFixed(2)} ${unit}`, "Yield effect"]}
                  contentStyle={{ borderRadius: "10px", fontSize: "12px", border: "1px solid #e2e8f0" }}
                />
                <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.impact >= 0 ? "#2f7c4e" : "#e11d48"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 space-y-1.5">
            {gap.ranked_factors
              .filter((f) => f.estimated_impact_on_yield !== 0)
              .map((f, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-slate-700">
                  <ArrowRight className="w-3.5 h-3.5 text-forest-600 flex-shrink-0 mt-0.5" />
                  <span>{f.explanation}</span>
                </div>
              ))}
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-600">
          None of the measured soil, weather or input differences changes the model's prediction noticeably.
        </p>
      )}

      <div className="flex items-start gap-2 text-[11px] text-slate-500 bg-slate-50 rounded-lg p-3">
        <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
        <span>{gap.caveat}</span>
      </div>
    </div>
  );
};

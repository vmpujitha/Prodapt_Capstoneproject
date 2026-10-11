import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { Activity, Sliders } from "lucide-react";
import { ImpactAnalysisResponse } from "../types/api";

interface Props {
  impact: ImpactAnalysisResponse | null | undefined;
}

export const ImpactCharts: React.FC<Props> = ({ impact }) => {
  if (!impact) return null;

  const importanceData = (impact.global_importance || []).slice(0, 7);
  const unit = impact.yield_unit || "t/ha";
  const sensitivityData = (impact.what_if_sensitivities || []).map((s) => ({
    name: s.factor.split("(")[0].trim(),
    delta: s.sensitivity_delta,
    plus: s.plus_20_predicted_yield,
    minus: s.minus_20_predicted_yield,
    baseline: s.current_predicted_yield
  }));

  return (
    <div className="space-y-2">
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 1. Global Feature Importance */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
            <Activity className="w-4 h-4 text-forest-600" />
            Global Yield Drivers for {impact.crop}
          </h4>
          <span className="text-[11px] text-slate-400">Permutation Importance (%)</span>
        </div>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={importanceData} layout="vertical" margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
              <XAxis type="number" domain={[0, "auto"]} unit="%" tick={{ fontSize: 10, fill: "#64748b" }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: "#334155" }} width={120} />
              <Tooltip
                formatter={(val: any) => [`${val}%`, "Relative Influence"]}
                contentStyle={{ borderRadius: "10px", fontSize: "12px", border: "1px solid #e2e8f0" }}
              />
              <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                {importanceData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={index === 0 ? "#2f7c4e" : index === 1 ? "#3f9a63" : "#97d1ac"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. What-If Sensitivity Tornado Chart */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
            <Sliders className="w-4 h-4 text-amber-600" />
            What-If Sensitivity (±20% Input Perturbation)
          </h4>
          <span className="text-[11px] text-slate-400">Δ Predicted yield ({unit}), -20% → +20%</span>
        </div>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sensitivityData} layout="vertical" margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
              <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "#334155" }} width={100} />
              <ReferenceLine x={0} stroke="#94a3b8" />
              <Tooltip
                formatter={(val: any) => [`${Number(val) > 0 ? "+" : ""}${Number(val).toFixed(3)} ${unit}`, "Yield Delta"]}
                contentStyle={{ borderRadius: "10px", fontSize: "12px", border: "1px solid #e2e8f0" }}
              />
              <Bar dataKey="delta" radius={[4, 4, 4, 4]}>
                {sensitivityData.map((entry, index) => (
                  <Cell key={`sens-${index}`} fill={entry.delta >= 0 ? "#2f7c4e" : "#e11d48"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
      <p className="text-[11px] text-slate-500 px-1">
        {impact.summary}
      </p>
    </div>
  );
};

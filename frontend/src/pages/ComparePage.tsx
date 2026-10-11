import React, { useState } from "react";
import { Scale, TrendingUp, ArrowRight, RefreshCw, BarChart3, AlertCircle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { api } from "../api/client";
import { CompareResponse, FarmProfile } from "../types/api";

interface Props {
  metadataOptions: any;
}

export const ComparePage: React.FC<Props> = ({ metadataOptions }) => {
  const [crop, setCrop] = useState("Wheat");
  const [season, setSeason] = useState("Rabi");
  const [stateName, setStateName] = useState("Punjab");

  const [farm1, setFarm1] = useState<FarmProfile>({
    name: "Farm 1 (High Yield)",
    crop: "Wheat",
    season: "Rabi",
    state: "Punjab",
    n: 110,
    p: 55,
    k: 45,
    ph: 6.8,
    temperature: 18.5,
    rainfall: 380,
    humidity: 60,
    fertilizer: 170,
    pesticide: 0.35,
    actual_yield: 4.2
  });

  const [farm2, setFarm2] = useState<FarmProfile>({
    name: "Farm 2 (Lagging Yield)",
    crop: "Wheat",
    season: "Rabi",
    state: "Punjab",
    n: 60,
    p: 30,
    k: 25,
    ph: 6.8,
    temperature: 18.5,
    rainfall: 200,
    humidity: 60,
    fertilizer: 110,
    pesticide: 0.2,
    actual_yield: 2.8
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    try {
      const p1 = { ...farm1, crop, season, state: stateName };
      const p2 = { ...farm2, crop, season, state: stateName };
      const res = await api.compareFarms({ profiles: [p1, p2] });
      setComparison(res);
    } catch (err: any) {
      setError(err.message || "Failed to compute farm comparison.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = (comparison?.ranked_factors || []).map((f) => ({
    factor: f.factor,
    impact: f.estimated_impact_on_yield,
    diff: f.difference,
    explanation: f.explanation
  }));

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs">
        <div className="flex items-center gap-2 text-forest-700 font-semibold text-xs uppercase tracking-wider mb-2">
          <Scale className="w-4 h-4 text-forest-600" />
          Yield Gap Diagnostics & Farm Comparison
        </div>
        <h2 className="text-xl font-bold text-slate-900">Compare Farm Profiles & Deconstruct Yield Gaps</h2>
        <p className="text-xs text-slate-500 mt-1">
          Input two agricultural profiles for the same crop to isolate which soil nutrients or weather variables drive the productivity gap.
        </p>

        {/* Global Common Attributes */}
        <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-slate-100 text-xs">
          <div>
            <label className="block text-slate-600 font-medium mb-1">Crop</label>
            <select
              value={crop}
              onChange={(e) => setCrop(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-slate-800"
            >
              {(metadataOptions?.crops || ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"]).map((c: string) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-slate-600 font-medium mb-1">Season</label>
            <select
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-slate-800"
            >
              {(metadataOptions?.seasons || ["Rabi", "Kharif", "Whole Year"]).map((s: string) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-slate-600 font-medium mb-1">State</label>
            <select
              value={stateName}
              onChange={(e) => setStateName(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-slate-800"
            >
              {(metadataOptions?.states || ["Punjab", "Haryana", "Uttar Pradesh"]).map((st: string) => (
                <option key={st} value={st}>
                  {st}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Profiles Side-by-Side Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Farm 1 */}
        <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-slate-900 text-sm">Farm Profile 1</h3>
            <span className="text-xs px-2.5 py-0.5 bg-forest-50 text-forest-700 font-semibold rounded-full border border-forest-200">
              Benchmark
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="text-slate-500 font-medium">Nitrogen (N kg/ha)</label>
              <input
                type="number"
                value={farm1.n}
                onChange={(e) => setFarm1({ ...farm1, n: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Phosphorus (P kg/ha)</label>
              <input
                type="number"
                value={farm1.p}
                onChange={(e) => setFarm1({ ...farm1, p: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Potassium (K kg/ha)</label>
              <input
                type="number"
                value={farm1.k}
                onChange={(e) => setFarm1({ ...farm1, k: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Soil pH</label>
              <input
                type="number"
                step="0.1"
                value={farm1.ph}
                onChange={(e) => setFarm1({ ...farm1, ph: parseFloat(e.target.value) || 7 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Rainfall (mm)</label>
              <input
                type="number"
                value={farm1.rainfall}
                onChange={(e) => setFarm1({ ...farm1, rainfall: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Fertilizer (kg/ha)</label>
              <input
                type="number"
                value={farm1.fertilizer}
                onChange={(e) => setFarm1({ ...farm1, fertilizer: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
          </div>
        </div>

        {/* Farm 2 */}
        <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-slate-900 text-sm">Farm Profile 2</h3>
            <span className="text-xs px-2.5 py-0.5 bg-amber-50 text-amber-700 font-semibold rounded-full border border-amber-200">
              Comparison Target
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="text-slate-500 font-medium">Nitrogen (N kg/ha)</label>
              <input
                type="number"
                value={farm2.n}
                onChange={(e) => setFarm2({ ...farm2, n: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Phosphorus (P kg/ha)</label>
              <input
                type="number"
                value={farm2.p}
                onChange={(e) => setFarm2({ ...farm2, p: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Potassium (K kg/ha)</label>
              <input
                type="number"
                value={farm2.k}
                onChange={(e) => setFarm2({ ...farm2, k: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Soil pH</label>
              <input
                type="number"
                step="0.1"
                value={farm2.ph}
                onChange={(e) => setFarm2({ ...farm2, ph: parseFloat(e.target.value) || 7 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Rainfall (mm)</label>
              <input
                type="number"
                value={farm2.rainfall}
                onChange={(e) => setFarm2({ ...farm2, rainfall: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
            <div>
              <label className="text-slate-500 font-medium">Fertilizer (kg/ha)</label>
              <input
                type="number"
                value={farm2.fertilizer}
                onChange={(e) => setFarm2({ ...farm2, fertilizer: parseFloat(e.target.value) || 0 })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2 mt-1"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center">
        <button
          onClick={handleCompare}
          disabled={loading}
          className="bg-forest-700 hover:bg-forest-800 text-white font-semibold py-3 px-8 rounded-2xl shadow-md transition flex items-center gap-2 cursor-pointer disabled:opacity-50"
        >
          {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <TrendingUp className="w-5 h-5" />}
          <span>{loading ? "Analyzing Factors..." : "Diagnose Yield Gap"}</span>
        </button>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-2xl flex items-center gap-3 text-xs">
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Summary Box */}
          <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs flex items-center justify-between flex-wrap gap-4">
            <div>
              <span className="text-xs font-semibold text-forest-700 uppercase tracking-wider block">Diagnostic Summary</span>
              <p className="text-sm font-bold text-slate-900 mt-1">{comparison.comparison_summary}</p>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-500 block">Total Yield Gap</span>
              <span
                className={`text-2xl font-extrabold ${
                  comparison.yield_gap >= 0 ? "text-forest-700" : "text-rose-600"
                }`}
              >
                {comparison.yield_gap > 0 ? "+" : ""}
                {comparison.yield_gap.toFixed(2)} {comparison.yield_unit || "t/ha"}
              </span>
            </div>
          </div>

          {/* Factor Breakdown Chart */}
          <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs">
            <h4 className="font-bold text-slate-900 text-sm mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-forest-600" />
              Factor-by-Factor Yield Impact Contribution
            </h4>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30, top: 10, bottom: 10 }}>
                  <XAxis type="number" unit={` ${comparison.yield_unit || "t/ha"}`} tick={{ fontSize: 10, fill: "#64748b" }} />
                  <YAxis type="category" dataKey="factor" tick={{ fontSize: 10, fill: "#334155" }} width={120} />
                  <ReferenceLine x={0} stroke="#94a3b8" />
                  <Tooltip
                    formatter={(val: any) => [`${Number(val) > 0 ? "+" : ""}${Number(val).toFixed(2)} ${comparison.yield_unit || "t/ha"}`, "Yield Shift"]}
                    contentStyle={{ borderRadius: "10px", fontSize: "12px", border: "1px solid #e2e8f0" }}
                  />
                  <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`comp-${index}`} fill={entry.impact >= 0 ? "#2f7c4e" : "#e11d48"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* List of Explanations */}
            <div className="mt-4 pt-4 border-t border-slate-100 space-y-2">
              {comparison.ranked_factors.map((f, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-slate-700">
                  <ArrowRight className="w-3.5 h-3.5 text-forest-600 flex-shrink-0 mt-0.5" />
                  <span>{f.explanation}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

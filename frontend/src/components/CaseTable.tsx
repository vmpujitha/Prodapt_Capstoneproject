import React from "react";
import { Table, ArrowUpDown, Info } from "lucide-react";
import { CaseDocument } from "../types/api";

interface Props {
  cases: CaseDocument[];
  onSelectCase: (caseId: string) => void;
}

export const CaseTable: React.FC<Props> = ({ cases, onSelectCase }) => {
  if (!cases || cases.length === 0) {
    return (
      <div className="p-8 text-center text-slate-400 text-xs bg-white rounded-2xl border border-slate-200">
        No similar historical cases retrieved.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <h3 className="font-bold text-slate-900 text-sm flex items-center gap-2">
          <Table className="w-4 h-4 text-forest-600" />
          Ranked Historical Evidence & Score Breakdown
        </h3>
        <span className="text-xs text-slate-500">{cases.length} cases retrieved</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-slate-50/80 text-slate-600 font-semibold border-b border-slate-200">
              <th className="py-3 px-4">Case ID</th>
              <th className="py-3 px-4">Crop</th>
              <th className="py-3 px-4">State & Season</th>
              <th className="py-3 px-4">Soil N-P-K (pH)</th>
              <th className="py-3 px-4">Weather (Rain/Temp)</th>
              <th className="py-3 px-4">Yield</th>
              <th className="py-3 px-4">Ranking Breakdown</th>
              <th className="py-3 px-4 text-right">Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {cases.map((c) => {
              const m = c.metadata;
              const s = c.scores;
              return (
                <tr
                  key={c.case_id}
                  onClick={() => onSelectCase(c.case_id)}
                  className="hover:bg-forest-50/40 cursor-pointer transition-colors"
                >
                  <td className="py-3 px-4 font-mono font-semibold text-forest-800">
                    {c.case_id}
                  </td>
                  <td className="py-3 px-4 font-bold text-slate-900">
                    {m.crop}
                  </td>
                  <td className="py-3 px-4 text-slate-600">
                    {m.state}, {m.season} '{String(m.crop_year).slice(-2)}
                  </td>
                  <td className="py-3 px-4 text-slate-600">
                    N:{m.n?.toFixed(0)} P:{m.p?.toFixed(0)} K:{m.k?.toFixed(0)} (pH {m.ph?.toFixed(1)})
                  </td>
                  <td className="py-3 px-4 text-slate-600">
                    {m.rainfall?.toFixed(0)}mm / {m.temperature?.toFixed(0)}°C
                  </td>
                  <td className="py-3 px-4">
                    <span className="font-bold text-slate-900">{m.yield?.toFixed(2)}</span>
                    <span className="text-[10px] text-slate-400 block">Top {Math.round(100 - (m.yield_percentile_within_crop || 50))}%</span>
                  </td>
                  {/* Score breakdown mini-bars */}
                  <td className="py-3 px-4 w-44">
                    <div className="space-y-1 text-[10px]">
                      <div className="flex justify-between text-slate-400">
                        <span>Retr/Prox</span>
                        <span>{(s.retrieval * 100).toFixed(0)}% / {(s.proximity * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden flex">
                        <div style={{ width: `${s.retrieval * 50}%` }} className="bg-forest-500 h-full" />
                        <div style={{ width: `${s.proximity * 50}%` }} className="bg-emerald-400 h-full" />
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className="font-bold text-forest-800 text-sm">
                      {s.final.toFixed(2)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

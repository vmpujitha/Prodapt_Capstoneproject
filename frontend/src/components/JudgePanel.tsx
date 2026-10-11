import React from "react";
import { CheckCircle2, XCircle, Scale, Star } from "lucide-react";
import { JudgeVerdict } from "../types/api";

interface Props {
  judge: JudgeVerdict | null | undefined;
}

export const JudgePanel: React.FC<Props> = ({ judge }) => {
  if (!judge) return null;

  const metrics = [
    { label: "Grounded in Cases", score: judge.grounded },
    { label: "Data Consistency", score: judge.consistent_with_data },
    { label: "Actionability", score: judge.actionable },
    { label: "Safety / Non-Hazardous", score: judge.safe }
  ];

  const avgScore = (
    (judge.grounded + judge.consistent_with_data + judge.actionable + judge.safe) / 4
  ).toFixed(1);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
            <Scale className="w-4 h-4" />
          </div>
          <div>
            <h4 className="font-bold text-slate-900 text-sm">Advisory Quality & Validation Judge</h4>
            <p className="text-xs text-slate-500">Automated multi-criteria evaluation of the generated advisory</p>
          </div>
        </div>

        {/* Pass / Fail Status Badge */}
        <div
          className={`flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-bold border ${
            judge.passed
              ? "bg-emerald-50 text-emerald-800 border-emerald-200"
              : "bg-rose-50 text-rose-800 border-rose-200"
          }`}
        >
          {judge.passed ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>PASSED VALIDATION ({avgScore}/5.0)</span>
            </>
          ) : (
            <>
              <XCircle className="w-4 h-4 text-rose-600" />
              <span>REVISION NEEDED</span>
            </>
          )}
        </div>
      </div>

      {/* 4-Criteria Score Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
        {metrics.map((m) => (
          <div key={m.label} className="bg-slate-50 p-2.5 rounded-xl border border-slate-100 text-xs">
            <span className="text-slate-500 block text-[11px] font-medium">{m.label}</span>
            <div className="flex items-center gap-1 mt-1 font-bold text-slate-900">
              <span className="text-sm">{m.score}/5</span>
              <div className="flex text-amber-400">
                {[...Array(5)].map((_, i) => (
                  <Star
                    key={i}
                    className={`w-3 h-3 ${i < m.score ? "fill-amber-400 text-amber-400" : "text-slate-200"}`}
                  />
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Critique text */}
      <div className="text-xs text-slate-600 bg-slate-50/70 p-3 rounded-xl border border-slate-100 italic">
        <span className="font-semibold text-slate-700 not-italic mr-1.5">Judge Critique:</span>
        {judge.critique}
      </div>
    </div>
  );
};

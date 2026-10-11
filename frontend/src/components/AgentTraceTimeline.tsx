import React from "react";
import { GitCommit, Clock, Check } from "lucide-react";
import { TraceStep } from "../types/api";

interface Props {
  trace: TraceStep[] | undefined;
}

export const AgentTraceTimeline: React.FC<Props> = ({ trace }) => {
  if (!trace || trace.length === 0) return null;

  const totalMs = trace.reduce((acc, t) => acc + t.ms, 0);

  const formatNodeName = (name: string) => {
    return name
      .replace("_node", "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase());
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
          <GitCommit className="w-4 h-4 text-forest-600" />
          LangGraph Multi-Agent Workflow Trace
        </h4>
        <div className="flex items-center gap-1 text-xs text-slate-500 font-medium">
          <Clock className="w-3.5 h-3.5" />
          <span>Total Latency: <strong>{totalMs} ms</strong></span>
        </div>
      </div>

      {/* Horizontal / Wrapped Step Flow */}
      <div className="space-y-2">
        {trace.map((step, idx) => (
          <div
            key={idx}
            className="flex items-start gap-3 p-2.5 rounded-xl bg-slate-50/70 border border-slate-100 text-xs"
          >
            <div className="w-5 h-5 rounded-full bg-forest-100 text-forest-700 flex items-center justify-center flex-shrink-0 mt-0.5 font-bold text-[10px]">
              {idx + 1}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-900">{formatNodeName(step.node)}</span>
                <span className="text-[11px] font-mono text-slate-400 font-medium">{step.ms} ms</span>
              </div>
              <p className="text-slate-600 truncate text-[11px] mt-0.5">{step.summary}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

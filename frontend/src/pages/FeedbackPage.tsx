import React, { useState, useEffect } from "react";
import { MessageSquare, ThumbsUp, ThumbsDown, Award, Sparkles, RefreshCw, HelpCircle } from "lucide-react";
import { api } from "../api/client";
import { FeedbackStats } from "../types/api";

export const FeedbackPage: React.FC = () => {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(false);

  const loadStats = async () => {
    setLoading(true);
    try {
      const s = await api.getFeedbackStats();
      setStats(s);
    } catch (err) {
      console.error("Load feedback stats error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 text-forest-700 font-semibold text-xs uppercase tracking-wider mb-1">
            <MessageSquare className="w-4 h-4 text-forest-600" />
            Continuous Learning & Evaluation
          </div>
          <h2 className="text-xl font-bold text-slate-900">Feedback Loop & Dynamic Re-Ranking Analytics</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Farmer ratings and agronomist annotations feed directly into Laplace-smoothed case prioritization.
          </p>
        </div>

        <button
          onClick={loadStats}
          disabled={loading}
          className="p-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition"
          title="Refresh stats"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-semibold text-slate-400 block">Total Evaluations Logged</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">
            {stats?.total_feedback || 0}
          </div>
          <span className="text-[11px] text-slate-500">Stored in persistent SQLite DB</span>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-semibold text-emerald-600 flex items-center gap-1">
            <ThumbsUp className="w-3.5 h-3.5" /> Positive Ratings
          </span>
          <div className="text-2xl font-extrabold text-emerald-800 mt-1">
            {stats?.positive_count || 0}
          </div>
          <span className="text-[11px] text-slate-500">
            {stats?.total_feedback
              ? `${Math.round((stats.positive_count / stats.total_feedback) * 100)}% satisfaction`
              : "No feedback recorded yet"}
          </span>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-2xs">
          <span className="text-xs font-semibold text-rose-600 flex items-center gap-1">
            <ThumbsDown className="w-3.5 h-3.5" /> Negative Ratings
          </span>
          <div className="text-2xl font-extrabold text-rose-800 mt-1">
            {stats?.negative_count || 0}
          </div>
          <span className="text-[11px] text-slate-500">Triggers re-ranking dampening</span>
        </div>
      </div>

      {/* Top Helpful Historical Cases */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-xs">
        <h3 className="font-bold text-slate-900 text-sm mb-4 flex items-center gap-2">
          <Award className="w-4 h-4 text-forest-600" />
          Top Rated Helpful Historical Cases
        </h3>

        {stats?.top_helpful_cases && stats.top_helpful_cases.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {stats.top_helpful_cases.map((c) => (
              <div key={c.case_id} className="p-3 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between text-xs">
                <span className="font-mono font-bold text-forest-800">{c.case_id}</span>
                <span className="text-slate-600 font-semibold">{c.helpful_count} helpful votes</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center text-slate-400 text-xs bg-slate-50/50 rounded-2xl border border-slate-100">
            No specific cases voted as most helpful yet. Submit feedback on the Advisor page to see stats here!
          </div>
        )}

        {/* Technical Notice */}
        <div className="mt-6 p-4 bg-slate-50 rounded-2xl border border-slate-200 text-xs text-slate-600 space-y-1">
          <div className="font-semibold text-slate-800 flex items-center gap-1.5">
            <HelpCircle className="w-3.5 h-3.5 text-forest-600" />
            Continuous Improvement Mechanism Note
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            User feedback directly modifies the <code className="font-mono bg-slate-200/60 px-1 py-0.2 rounded text-slate-800">feedback_boost</code> parameter (bounded to [-0.1, +0.1] with Laplace smoothing) applied in the multi-component ranking formula. This provides real-time ranking adjustments without expensive offline retraining.
          </p>
        </div>
      </div>
    </div>
  );
};

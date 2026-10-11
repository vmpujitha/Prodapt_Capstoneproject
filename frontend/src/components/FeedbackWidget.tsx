import React, { useState } from "react";
import { ThumbsUp, ThumbsDown, CheckCircle, MessageSquare } from "lucide-react";
import { api } from "../api/client";
import { CaseDocument } from "../types/api";

interface Props {
  query: string;
  cases: CaseDocument[];
  topCrop?: string;
}

export const FeedbackWidget: React.FC<Props> = ({ query, cases, topCrop }) => {
  const [rating, setRating] = useState<number | null>(null);
  const [comment, setComment] = useState("");
  const [selectedCases, setSelectedCases] = useState<string[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const toggleCase = (cid: string) => {
    if (selectedCases.includes(cid)) {
      setSelectedCases(selectedCases.filter((c) => c !== cid));
    } else {
      setSelectedCases([...selectedCases, cid]);
    }
  };

  const handleSubmit = async () => {
    if (rating === null) return;
    setSubmitting(true);
    try {
      await api.submitFeedback({
        query,
        case_ids: cases.map((c) => c.case_id),
        recommended_crop: topCrop,
        rating,
        comment: comment.trim() || undefined,
        helpful_case_ids: selectedCases
      });
      setSubmitted(true);
    } catch (err) {
      console.error("Feedback submit error:", err);
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="bg-forest-50 border border-forest-200 rounded-2xl p-4 flex items-center gap-3 text-xs text-forest-800">
        <CheckCircle className="w-5 h-5 text-forest-600 flex-shrink-0" />
        <div>
          <span className="font-bold">Thank you for your feedback!</span>
          <p className="text-forest-700 mt-0.5">
            Your evaluation has been stored and will continuously refine historical case prioritization.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs text-xs">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="font-bold text-slate-900 text-sm">Was this advisory recommendation helpful?</h4>
          <p className="text-slate-500 text-xs">Help continuous learning by rating historical case relevance.</p>
        </div>

        {/* Thumbs Up / Down */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRating(1)}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-xl border transition ${
              rating === 1
                ? "bg-forest-50 border-forest-400 text-forest-800 font-bold"
                : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
            }`}
          >
            <ThumbsUp className="w-3.5 h-3.5 text-forest-600" />
            <span>Yes</span>
          </button>
          <button
            onClick={() => setRating(-1)}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-xl border transition ${
              rating === -1
                ? "bg-rose-50 border-rose-400 text-rose-800 font-bold"
                : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
            }`}
          >
            <ThumbsDown className="w-3.5 h-3.5 text-rose-600" />
            <span>No</span>
          </button>
        </div>
      </div>

      {rating !== null && (
        <div className="mt-3 pt-3 border-t border-slate-100 space-y-3 animate-in fade-in duration-150">
          {/* Helpful Cases Select */}
          {cases.length > 0 && (
            <div>
              <span className="font-semibold text-slate-700 block mb-1.5">
                Which specific historical cases provided the best evidence?
              </span>
              <div className="flex flex-wrap gap-1.5">
                {cases.map((c) => {
                  const isSel = selectedCases.includes(c.case_id);
                  return (
                    <button
                      key={c.case_id}
                      type="button"
                      onClick={() => toggleCase(c.case_id)}
                      className={`font-mono text-xs px-2.5 py-1 rounded-lg border transition ${
                        isSel
                          ? "bg-forest-600 border-forest-600 text-white font-bold"
                          : "bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100"
                      }`}
                    >
                      {c.case_id} ({c.metadata.crop})
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Comment */}
          <div>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Optional field notes or agronomist comment..."
              rows={2}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-forest-500"
            />
          </div>

          <div className="flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="bg-forest-700 hover:bg-forest-800 text-white font-semibold py-1.5 px-4 rounded-xl shadow-xs transition disabled:opacity-50"
            >
              {submitting ? "Submitting..." : "Submit Feedback"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

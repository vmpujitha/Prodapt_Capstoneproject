import React from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  isSynthetic: boolean;
}

export const SyntheticBanner: React.FC<Props> = ({ isSynthetic }) => {
  if (!isSynthetic) return null;

  return (
    <div className="bg-amber-500/15 border-b border-amber-500/30 text-amber-900 px-4 py-2 flex items-center justify-between text-xs font-medium">
      <div className="flex items-center gap-2 max-w-7xl mx-auto w-full">
        <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
        <span>
          <strong className="font-semibold uppercase tracking-wider text-amber-800 mr-1.5">Synthetic Dataset Active:</strong>
          Running on realistic synthetic agricultural data generated according to Kaggle statistical distributions. Place real dataset at <code className="bg-amber-200/60 px-1 py-0.5 rounded text-amber-950 font-mono">data/raw/crop_yield.csv</code> for production deployment.
        </span>
      </div>
    </div>
  );
};

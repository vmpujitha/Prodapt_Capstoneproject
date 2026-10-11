import React from "react";
import { X, FileText, MapPin, Calendar, Droplets, Thermometer, FlaskConical } from "lucide-react";
import { CaseDocument } from "../types/api";
import { yieldUnit } from "../utils/units";

interface Props {
  caseDoc: CaseDocument | null;
  onClose: () => void;
}

export const CaseDetailsModal: React.FC<Props> = ({ caseDoc, onClose }) => {
  if (!caseDoc) return null;
  const m = caseDoc.metadata;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 max-w-xl w-full p-6 relative overflow-hidden animate-in zoom-in-95 duration-150">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-forest-100 text-forest-800 flex items-center justify-center font-bold">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-slate-900">{caseDoc.case_id}</h3>
              <span className="text-xs font-semibold px-2 py-0.5 bg-forest-50 text-forest-700 rounded-md border border-forest-200">
                {m.crop}
              </span>
            </div>
            <p className="text-xs text-slate-500 flex items-center gap-2 mt-0.5">
              <span className="flex items-center gap-1">
                <MapPin className="w-3 h-3" /> {m.state}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" /> {m.season} {m.crop_year}
              </span>
            </p>
          </div>
        </div>

        {/* Narrative Box */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 mb-4 text-xs text-slate-700 leading-relaxed italic">
          "{caseDoc.narrative}"
        </div>

        {/* Condition Metric Grid */}
        <div className="grid grid-cols-2 gap-3 text-xs mb-4">
          {/* Soil */}
          <div className="p-3 bg-slate-50/70 rounded-xl border border-slate-100">
            <div className="font-semibold text-slate-800 flex items-center gap-1.5 mb-2">
              <FlaskConical className="w-3.5 h-3.5 text-forest-600" />
              Soil Nutrients & pH
            </div>
            <div className="space-y-1 text-slate-600">
              <div className="flex justify-between">
                <span>Soil pH:</span>
                <span className="font-semibold text-slate-900">{m.ph} ({m.ph_class})</span>
              </div>
              <div className="flex justify-between">
                <span>Nitrogen (N):</span>
                <span className="font-semibold text-slate-900">{m.n} kg/ha ({m.n_level})</span>
              </div>
              <div className="flex justify-between">
                <span>Phosphorus (P):</span>
                <span className="font-semibold text-slate-900">{m.p} kg/ha ({m.p_level})</span>
              </div>
              <div className="flex justify-between">
                <span>Potassium (K):</span>
                <span className="font-semibold text-slate-900">{m.k} kg/ha ({m.k_level})</span>
              </div>
            </div>
          </div>

          {/* Weather */}
          <div className="p-3 bg-slate-50/70 rounded-xl border border-slate-100">
            <div className="font-semibold text-slate-800 flex items-center gap-1.5 mb-2">
              <Thermometer className="w-3.5 h-3.5 text-amber-600" />
              Climate & Weather
            </div>
            <div className="space-y-1 text-slate-600">
              <div className="flex justify-between">
                <span>Temperature:</span>
                <span className="font-semibold text-slate-900">{m.temperature}°C ({m.temp_class})</span>
              </div>
              <div className="flex justify-between">
                <span>Rainfall:</span>
                <span className="font-semibold text-slate-900">{m.rainfall} mm ({m.rainfall_class})</span>
              </div>
              <div className="flex justify-between">
                <span>Humidity:</span>
                <span className="font-semibold text-slate-900">{m.humidity}% ({m.humidity_class})</span>
              </div>
              <div className="flex justify-between">
                <span>Fertilizer:</span>
                <span className="font-semibold text-slate-900">{m.fertilizer} kg/ha</span>
              </div>
            </div>
          </div>
        </div>

        {/* Productivity Summary */}
        <div className="p-3 bg-forest-50/50 rounded-xl border border-forest-100 flex items-center justify-between text-xs">
          <div>
            <span className="text-slate-500 block">Cultivated Land</span>
            <span className="font-bold text-slate-900">{m.area} ha</span>
          </div>
          <div>
            <span className="text-slate-500 block">Total Production</span>
            <span className="font-bold text-slate-900">{m.production} tonnes</span>
          </div>
          <div>
            <span className="text-slate-500 block">Actual Harvest Yield</span>
            <span className="font-bold text-forest-800 text-sm">{Number(m.yield).toFixed(2)} {yieldUnit(m.crop)}</span>
          </div>
          <div>
            <span className="text-slate-500 block">Yield Percentile</span>
            <span className="font-bold text-forest-800">{m.yield_percentile_within_crop}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export interface ProviderStatus {
  provider: string;
  available: boolean;
  reason?: string | null;
  chat_models: string[];
  embedding_models: string[];
  default_chat_model?: string | null;
  default_embedding_model?: string | null;
}

export interface SoilCondition {
  n?: number | null;
  p?: number | null;
  k?: number | null;
  ph?: number | null;
  n_level?: string | null;
  p_level?: string | null;
  k_level?: string | null;
  ph_class?: string | null;
}

export interface WeatherCondition {
  temperature?: number | null;
  rainfall?: number | null;
  humidity?: number | null;
  temp_class?: string | null;
  rainfall_class?: string | null;
  humidity_class?: string | null;
}

export interface QueryClassification {
  intent: string;
  crop?: string | null;
  season?: string | null;
  state?: string | null;
  soil: SoilCondition;
  weather: WeatherCondition;
  confidence: number;
}

export interface ScoreBreakdown {
  retrieval: number;
  proximity: number;
  yield_pct: number;
  recency: number;
  feedback: number;
  final: number;
}

export interface CaseDocument {
  case_id: string;
  narrative: string;
  metadata: Record<string, any>;
  scores: ScoreBreakdown;
}

export interface RiskFactor {
  factor: string;
  impact: string;
  description: string;
}

export interface RiskAssessment {
  level: string;
  probability: number;
  factors: RiskFactor[];
}

export interface YieldPredictionInterval {
  value: number;
  low: number;
  high: number;
  unit?: string;
}

export interface CropRecommendation {
  crop: string;
  score: number;
  median_yield: number;
  mean_yield: number;
  yield_percentile: number;
  similarity: number;
  case_count: number;
  supporting_cases: string[];
  risk?: RiskAssessment | null;
  predicted_yield?: YieldPredictionInterval | null;
  score_breakdown?: Record<string, number> | null;
}

export interface GlobalImportance {
  feature: string;
  importance: number;
}

export interface SensitivityFactor {
  factor: string;
  baseline_value: number;
  current_predicted_yield: number;
  plus_20_predicted_yield: number;
  minus_20_predicted_yield: number;
  sensitivity_delta: number;
}

export interface ImpactAnalysisResponse {
  crop: string;
  yield_unit?: string;
  importance_method?: string;
  global_importance: GlobalImportance[];
  what_if_sensitivities: SensitivityFactor[];
  summary: string;
}

export interface JudgeVerdict {
  grounded: number;
  consistent_with_data: number;
  actionable: number;
  safe: number;
  passed: boolean;
  critique: string;
}

export interface TraceStep {
  node: string;
  ms: number;
  summary: string;
}

export interface MetadataFilters {
  crop?: string | null;
  season?: string | null;
  state?: string | null;
  crop_year_min?: number | null;
  crop_year_max?: number | null;
  ph_class?: string | null;
  n_level?: string | null;
  p_level?: string | null;
  k_level?: string | null;
  temp_class?: string | null;
  rainfall_class?: string | null;
  humidity_class?: string | null;
}

export interface PipelineQueryRequest {
  query: string;
  filters?: MetadataFilters;
  top_k?: number;
  mode?: "hybrid" | "semantic" | "keyword";
  llm?: { provider?: string | null; model?: string | null };
  embedding?: { provider?: string | null; model?: string | null };
}

export interface PipelineQueryResponse {
  classification: QueryClassification;
  similar_cases: CaseDocument[];
  recommendations: CropRecommendation[];
  impact_analysis?: ImpactAnalysisResponse | null;
  explanation: string;
  judge?: JudgeVerdict | null;
  agent_trace: TraceStep[];
  llm_used?: { provider?: string; model?: string } | null;
  fallback_triggered: boolean;
  degraded_mode: boolean;
  disclaimer: string;
  yield_gap?: YieldGapAnalysis | null;
  warnings?: string[];
  filters_applied?: Record<string, any>;
  search_mode_used?: string | null;
  out_of_scope?: boolean;
  is_synthetic_data?: boolean;
}

export interface FarmProfile {
  name: string;
  crop: string;
  season: string;
  state: string;
  n: number;
  p: number;
  k: number;
  ph: number;
  temperature: number;
  rainfall: number;
  humidity: number;
  fertilizer?: number;
  pesticide?: number;
  actual_yield?: number | null;
}

export interface CompareRequest {
  profiles: FarmProfile[];
}

export interface YieldGapFactor {
  factor: string;
  farm1_val: number;
  farm2_val: number;
  difference: number;
  estimated_impact_on_yield: number;
  direction: string;
  explanation: string;
}

export interface CompareResponse {
  crop: string;
  yield_unit?: string;
  profiles: Array<{ name: string; yield: number; conditions: Record<string, any>; yield_source?: string }>;
  yield_gap: number;
  ranked_factors: YieldGapFactor[];
  comparison_summary: string;
}

export interface FeedbackCreate {
  query: string;
  case_ids: string[];
  recommended_crop?: string | null;
  rating: number;
  comment?: string | null;
  helpful_case_ids: string[];
}

export interface FeedbackStats {
  total_feedback: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  top_helpful_cases: Array<{ case_id: string; helpful_count: number }>;
  crop_ratings: Record<string, Record<string, number>>;
}

export interface EvalSampleResult {
  id?: string | null;
  query: string;
  scenario: string;
  expected_crops: string[];
  expected_crops_resolved?: string[];
  retrieved_cases: string[];
  retrieved_crops?: string[];
  recommended_crops?: string[];
  hit_at_k: boolean;
  recommendation_hit?: boolean;
  judge_passed?: boolean | null;
  degraded_mode?: boolean;
  relevancy_score?: number | null;
  faithfulness_score?: number | null;
  contextual_relevancy_score?: number | null;
  contextual_precision_score?: number | null;
  geval_score?: number | null;
  metric_reasons?: Record<string, string>;
  latency_ms?: number | null;
  error?: string | null;
  reason?: string | null;
}

export interface EvalStatusResponse {
  run_id: string;
  status: string;
  progress: number;
  total_samples: number;
  completed_samples: number;
  results?: EvalSampleResult[] | null;
  aggregate_metrics?: Record<string, number | null> | null;
  deepeval_status?: string | null;
  judge_model?: string | null;
  is_synthetic_data?: boolean | null;
  created_at: string;
  completed_at?: string | null;
  error?: string | null;
}

export interface IndexJob {
  job_id: string;
  collection: string;
  status: "running" | "completed" | "failed";
  indexed: number;
  total: number;
  progress: number;
  error?: string | null;
}

export interface IndexStatus {
  collection: string;
  provider: string;
  model: string;
  exists: boolean;
  document_count: number;
  expected_documents: number;
  up_to_date: boolean;
  embeddings_available: boolean;
  reason?: string | null;
  job?: IndexJob | null;
}

export interface YieldGapGroup {
  name: string;
  case_count: number;
  median_yield: number;
  conditions: Record<string, any>;
  example_case_ids: string[];
  example_cases?: Array<{ case_id: string; narrative: string; metadata: Record<string, any> }>;
}

export interface YieldGapAnalysis {
  crop: string;
  season?: string | null;
  grouping: "state" | "yield_quartile";
  yield_unit: string;
  groups: YieldGapGroup[];
  yield_gap: number;
  ranked_factors: YieldGapFactor[];
  summary: string;
  caveat: string;
}

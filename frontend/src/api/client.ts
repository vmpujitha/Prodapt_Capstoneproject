import {
  ProviderStatus,
  PipelineQueryRequest,
  PipelineQueryResponse,
  CompareRequest,
  CompareResponse,
  FeedbackCreate,
  FeedbackStats,
  EvalStatusResponse,
  IndexJob,
  IndexStatus
} from "../types/api";

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

export class ApiError extends Error {
  constructor(message: string, public status: number, public code?: string) {
    super(message);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const headers = {
    "Content-Type": "application/json",
    ...(options?.headers || {})
  };

  const res = await fetch(url, { ...options, headers });
  if (!res.ok) {
    let errorDetail = "An error occurred";
    let code: string | undefined;
    try {
      const errJson = await res.json();
      errorDetail = errJson.error?.message || JSON.stringify(errJson);
      code = errJson.error?.code;
    } catch {
      errorDetail = res.statusText;
    }
    throw new ApiError(errorDetail, res.status, code);
  }

  return res.json();
}

export const api = {
  getHealth: () => request<any>("/health"),
  getProviders: () => request<ProviderStatus[]>("/providers"),
  getMetadataOptions: () => request<any>("/metadata/options"),
  
  queryPipeline: (req: PipelineQueryRequest) =>
    request<PipelineQueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(req)
    }),

  compareFarms: (req: CompareRequest) =>
    request<CompareResponse>("/compare", {
      method: "POST",
      body: JSON.stringify(req)
    }),

  submitFeedback: (fb: FeedbackCreate) =>
    request<{ id: number; status: string; message: string }>("/feedback", {
      method: "POST",
      body: JSON.stringify(fb)
    }),

  getFeedbackStats: () => request<FeedbackStats>("/feedback/stats"),

  triggerEval: (limit?: number, llm?: { provider?: string | null; model?: string | null }) =>
    request<{ run_id: string; status: string; total_samples: number; message: string }>("/eval/run", {
      method: "POST",
      body: JSON.stringify({ limit, llm })
    }),

  getEvalStatus: (runId: string) => request<EvalStatusResponse>(`/eval/status/${runId}`),
  getLatestEval: () => request<EvalStatusResponse>("/eval/latest"),

  getIndexStatus: (provider: string, model: string) =>
    request<IndexStatus>(
      `/admin/index/status?embedding_provider=${encodeURIComponent(provider)}&embedding_model=${encodeURIComponent(model)}`
    ),

  reindex: (provider: string, model: string) =>
    request<IndexJob>("/admin/reindex", {
      method: "POST",
      body: JSON.stringify({ embedding_provider: provider, embedding_model: model, force: true })
    }),

  getReindexStatus: (jobId: string) => request<IndexJob>(`/admin/reindex/status/${jobId}`),

  getGraphMermaid: () => request<{ mermaid: string }>("/graph")
};

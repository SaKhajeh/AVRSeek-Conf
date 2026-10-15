export type StageStatus = "pending" | "running" | "done" | "stubbed" | "skipped" | "error";

export interface PipelineStage {
  id: string;
  label: string;
  status: StageStatus;
  detail: string;
  data?: Record<string, unknown> | null;
}

export interface ChemicalEntity {
  name: string;
  role: string;
  amount?: string | null;
  smiles?: string | null;
  cas?: string | null;
}

export interface ReactionConditions {
  temperature: string;
  time: string;
  pressure: string;
  atmosphere: string;
  other: string;
}

export interface ReactionStep {
  step_index: number;
  step_label: string;
  reactants: ChemicalEntity[];
  reagents: ChemicalEntity[];
  catalysts: ChemicalEntity[];
  ligands: ChemicalEntity[];
  solvents: ChemicalEntity[];
  intermediates: ChemicalEntity[];
  products: ChemicalEntity[];
  conditions: ReactionConditions;
  yield_value: string;
  ee_or_dr: string;
  notes: string;
  evidence: string;
}

export interface SynthesisRoute {
  title: string;
  target_name: string;
  target_class: string;
  document_type: string;
  steps: ReactionStep[];
  warnings: string[];
}

export interface ParsedDocument {
  filename: string;
  page_count: number;
  text: string;
  figure_count: number;
  table_hints: string[];
}

export interface ExtractionResult {
  job_id: string;
  filename: string;
  stages: PipelineStage[];
  document?: ParsedDocument | null;
  route?: SynthesisRoute | null;
  raw_model_output?: string | null;
  error?: string | null;
  from_cache?: boolean;
  content_hash?: string | null;
  source_job_id?: string | null;
}

export interface HealthInfo {
  status: string;
  project: string;
  base_url: string;
  model_text: string;
  model_vlm: string;
  model_standardize: string;
  api_key_configured: boolean;
  port?: number;
  features?: string[];
}

export interface FileEstimate {
  filename: string;
  page_count: number;
  figure_count: number;
  table_hints: number;
  estimated_seconds: number;
  estimated_display: string;
  cached?: boolean;
}

export interface BatchEstimate {
  file_count: number;
  total_pages: number;
  total_figures: number;
  total_tables: number;
  estimated_seconds: number;
  estimated_minutes: number;
  estimated_display: string;
  note: string;
  cached_count?: number;
  per_file: FileEstimate[];
}

export interface FileLiveRow {
  filename: string;
  status: "queued" | "running" | "cached" | "ok" | "error";
  stageLabel: string;
  detail: string;
  pages?: number;
  figures?: number;
  steps?: number;
  warnings: string[];
  from_cache?: boolean;
  job_id?: string;
}

export interface BatchProgress {
  batchId: string | null;
  currentIndex: number;
  total: number;
  currentFile: string;
  estimatedDisplay: string;
  remainingDisplay: string;
  stageLabel: string;
  stageDetail: string;
  stageChecklist: PipelineStage[];
  completed: number;
  completedResults: ExtractionResult[];
  errors: { filename: string; detail: string }[];
  downloads: { json: string; excel: string } | null;
  finished: boolean;
}

import { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
} from 'recharts';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import rawModelParams from './ech_model_params.json';
import rawTrainSummary from './ech_train_summary.json';
import rawLockedTest from './ech_locked_test.json';

export interface FeatureCoefficient {
  name: string;
  label?: string;
  category?: string;
  coef: number;
  oddsRatio?: number;
  stdError?: number;
  zValue?: number;
  pValue?: number;
  mean?: number;
  min?: number;
  max?: number;
  type?: 'binary' | 'numeric' | 'categorical';
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  auc?: number;
  rocAuc?: number;
  logLoss?: number;
  brierScore?: number;
  nSamples?: number;
  positiveRate?: number;
  tp?: number;
  fp?: number;
  tn?: number;
  fn?: number;
}

export interface ThresholdPoint {
  threshold: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  tpr?: number;
  fpr?: number;
  specificity?: number;
}

export interface ModelParamsData {
  modelName?: string;
  targetDescription?: string;
  intercept: number;
  defaultThreshold?: number;
  features?: FeatureCoefficient[];
  coefficients?: Record<string, number>;
  thresholds?: ThresholdPoint[];
  metrics?: ModelMetrics;
}

export interface GroupSummaryRow {
  group: string;
  subGroup?: string;
  n?: number;
  informalityRate?: number;
  formalityRate?: number;
  avgPredictedProb?: number;
  [key: string]: string | number | undefined;
}

export interface TrainSummaryData {
  datasetName?: string;
  year?: number;
  nRows?: number;
  nFeatures?: number;
  informalityRate?: number;
  formalityRate?: number;
  metrics?: ModelMetrics;
  byGroup?: GroupSummaryRow[];
  byDepartment?: GroupSummaryRow[];
  byEducation?: GroupSummaryRow[];
  byAgeGroup?: GroupSummaryRow[];
  distributions?: Array<Record<string, string | number>>;
}

export interface LockedTestRecord {
  id?: string | number;
  actual?: number;
  actualInformal?: number;
  actualFormal?: number;
  predictedProb?: number;
  prob?: number;
  department?: string;
  education?: string;
  sex?: string;
  age?: number;
  [key: string]: string | number | boolean | null | undefined;
}

export interface CalibrationBin {
  bin: string;
  meanPredicted: number;
  observedRate: number;
  count: number;
}

export interface LockedTestData {
  nRows?: number;
  metrics?: ModelMetrics;
  thresholds?: ThresholdPoint[];
  rocCurve?: Array<{ fpr: number; tpr: number; threshold?: number }>;
  calibration?: CalibrationBin[];
  predictions?: LockedTestRecord[];
  rows?: LockedTestRecord[];
  summaryByDecile?: Array<Record<string, string | number>>;
}

const modelParams = rawModelParams as unknown as ModelParamsData;
const trainSummary = rawTrainSummary as unknown as TrainSummaryData;
const lockedTestData = rawLockedTest as unknown as LockedTestData;

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function formatPct(value: number | undefined, decimals = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  const scaled = value <= 1 && value >= 0 ? value * 100 : value;
  return `${scaled.toFixed(decimals)}%`;
}

function formatNum(value: number | undefined, decimals = 3): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return value.toFixed(decimals);
}

function formatInt(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return Math.round(value).toLocaleString('es-UY');
}

function humanizeFeatureName(raw: string): string {
  return raw
    .replace(/^factor\(([^)]+)\)/i, '$1: ')
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .trim();
}

type TabKey = 'overview' | 'coefficients' | 'evaluation' | 'simulator' | 'methodology';

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [threshold, setThreshold] = useState<number>(() => {
    if (typeof modelParams.defaultThreshold === 'number') {
      return clamp01(modelParams.defaultThreshold);
    }
    return 0.5;
  });
  const [featureSearch, setFeatureSearch] = useState<string>('');
  const [sortByAbsCoef, setSortByAbsCoef] = useState<boolean>(true);

  const normalizedFeatures = useMemo<FeatureCoefficient[]>(() => {
    if (Array.isArray(modelParams.features) && modelParams.features.length > 0) {
      return modelParams.features.map((f) => ({
        ...f,
        label: f.label || humanizeFeatureName(f.name),
        oddsRatio: typeof f.oddsRatio === 'number' ? f.oddsRatio : Math.exp(f.coef),
      }));
    }
    if (modelParams.coefficients && typeof modelParams.coefficients === 'object') {
      return Object.entries(modelParams.coefficients)
        .filter(([key]) => key.toLowerCase() !== '(intercept)' && key.toLowerCase() !== 'intercept')
        .map(([name, coef]) => ({
          name,
          label: humanizeFeatureName(name),
          coef: Number(coef),
          oddsRatio: Math.exp(Number(coef)),
        }));
    }
    return [];
  }, []);

  const interceptValue = useMemo<number>(() => {
    if (typeof modelParams.intercept === 'number' && !Number.isNaN(modelParams.intercept)) {
      return modelParams.intercept;
    }
    if (modelParams.coefficients) {
      const entry = Object.entries(modelParams.coefficients).find(
        ([k]) => k.toLowerCase() === '(intercept)' || k.toLowerCase() === 'intercept'
      );
      if (entry) return Number(entry[1]) || 0;
    }
    return 0;
  }, []);

  const [simInputs, setSimInputs] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    const list = Array.isArray(modelParams.features)
      ? modelParams.features
      : modelParams.coefficients
      ? Object.entries(modelParams.coefficients)
          .filter(([k]) => k.toLowerCase() !== '(intercept)' && k.toLowerCase() !== 'intercept')
          .map(([name]) => ({ name, coef: 0 }))
      : [];
    for (const f of list) {
      initial[f.name] = typeof f.mean === 'number' ? f.mean : 0;
    }
    return initial;
  });

  const testRecords = useMemo<LockedTestRecord[]>(() => {
    if (Array.isArray(lockedTestData.predictions)) return lockedTestData.predictions;
    if (Array.isArray(lockedTestData.rows)) return lockedTestData.rows;
    return [];
  }, []);

  const thresholdCurve = useMemo<ThresholdPoint[]>(() => {
    if (Array.isArray(lockedTestData.thresholds) && lockedTestData.thresholds.length > 0) {
      return lockedTestData.thresholds;
    }
    if (Array.isArray(modelParams.thresholds) && modelParams.thresholds.length > 0) {
      return modelParams.thresholds;
    }
    if (testRecords.length === 0) return [];

    const grid: ThresholdPoint[] = [];
    for (let t = 0.05; t <= 0.951; t += 0.05) {
      const thr = Number(t.toFixed(2));
      let tp = 0;
      let fp = 0;
      let tn = 0;
      let fn = 0;
      for (const r of testRecords) {
        const y =
          typeof r.actual === 'number'
            ? r.actual
            : typeof r.actualInformal === 'number'
            ? r.actualInformal
            : typeof r.actualFormal === 'number'
            ? r.actualFormal
            : 0;
        const p =
          typeof r.predictedProb === 'number'
            ? r.predictedProb
            : typeof r.prob === 'number'
            ? r.prob
            : 0;
        const pred = p >= thr ? 1 : 0;
        if (pred === 1 && y === 1) tp++;
        else if (pred === 1 && y === 0) fp++;
        else if (pred === 0 && y === 0) tn++;
        else fn++;
      }
      const total = tp + fp + tn + fn || 1;
      const accuracy = (tp + tn) / total;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      grid.push({
        threshold: thr,
        accuracy,
        precision,
        recall,
        f1,
        tpr: recall,
        fpr: 1 - specificity,
        specificity,
      });
    }
    return grid;
  }, [testRecords]);

  const activeThresholdMetrics = useMemo(() => {
    if (testRecords.length > 0) {
      let tp = 0;
      let fp = 0;
      let tn = 0;
      let fn = 0;
      for (const r of testRecords) {
        const y =
          typeof r.actual === 'number'
            ? r.actual
            : typeof r.actualInformal === 'number'
            ? r.actualInformal
            : typeof r.actualFormal === 'number'
            ? r.actualFormal
            : 0;
        const p =
          typeof r.predictedProb === 'number'
            ? r.predictedProb
            : typeof r.prob === 'number'
            ? r.prob
            : 0;
        const pred = p >= threshold ? 1 : 0;
        if (pred === 1 && y === 1) tp++;
        else if (pred === 1 && y === 0) fp++;
        else if (pred === 0 && y === 0) tn++;
        else fn++;
      }
      const total = tp + fp + tn + fn || 1;
      const accuracy = (tp + tn) / total;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return {
        tp,
        fp,
        tn,
        fn,
        accuracy,
        precision,
        recall,
        specificity,
        f1,
        auc:
          lockedTestData.metrics?.auc ??
          lockedTestData.metrics?.rocAuc ??
          modelParams.metrics?.auc ??
          modelParams.metrics?.rocAuc,
      };
    }

    if (thresholdCurve.length > 0) {
      let closest = thresholdCurve[0];
      for (const pt of thresholdCurve) {
        if (Math.abs(pt.threshold - threshold) < Math.abs(closest.threshold - threshold)) {
          closest = pt;
        }
      }
      return {
        tp: lockedTestData.metrics?.tp,
        fp: lockedTestData.metrics?.fp,
        tn: lockedTestData.metrics?.tn,
        fn: lockedTestData.metrics?.fn,
        accuracy: closest.accuracy ?? lockedTestData.metrics?.accuracy,
        precision: closest.precision ?? lockedTestData.metrics?.precision,
        recall: closest.recall ?? closest.tpr ?? lockedTestData.metrics?.recall,
        specificity: closest.specificity ?? (closest.fpr !== undefined ? 1 - closest.fpr : undefined),
        f1: closest.f1 ?? lockedTestData.metrics?.f1,
        auc:
          lockedTestData.metrics?.auc ??
          lockedTestData.metrics?.rocAuc ??
          modelParams.metrics?.auc ??
          modelParams.metrics?.rocAuc,
      };
    }

    return {
      tp: lockedTestData.metrics?.tp,
      fp: lockedTestData.metrics?.fp,
      tn: lockedTestData.metrics?.tn,
      fn: lockedTestData.metrics?.fn,
      accuracy: lockedTestData.metrics?.accuracy ?? modelParams.metrics?.accuracy,
      precision: lockedTestData.metrics?.precision ?? modelParams.metrics?.precision,
      recall: lockedTestData.metrics?.recall ?? modelParams.metrics?.recall,
      specificity: undefined,
      f1: lockedTestData.metrics?.f1 ?? modelParams.metrics?.f1,
      auc:
        lockedTestData.metrics?.auc ??
        lockedTestData.metrics?.rocAuc ??
        modelParams.metrics?.auc ??
        modelParams.metrics?.rocAuc,
    };
  }, [testRecords, threshold, thresholdCurve]);

  const rocData = useMemo(() => {
    if (Array.isArray(lockedTestData.rocCurve) && lockedTestData.rocCurve.length > 0) {
      return lockedTestData.rocCurve;
    }
    if (thresholdCurve.length > 0) {
      const pts = thresholdCurve
        .filter((d) => typeof d.fpr === 'number' && typeof d.tpr === 'number')
        .map((d) => ({
          fpr: Number((d.fpr as number).toFixed(4)),
          tpr: Number((d.tpr as number).toFixed(4)),
          threshold: d.threshold,
        }))
        .sort((a, b) => a.fpr - b.fpr);
      return [{ fpr: 0, tpr: 0, threshold: 1 }, ...pts, { fpr: 1, tpr: 1, threshold: 0 }];
    }
    return [];
  }, [thresholdCurve]);

  const calibrationData = useMemo<CalibrationBin[]>(() => {
    if (Array.isArray(lockedTestData.calibration) && lockedTestData.calibration.length > 0) {
      return lockedTestData.calibration;
    }
    if (Array.isArray(lockedTestData.summaryByDecile) && lockedTestData.summaryByDecile.length > 0) {
      return lockedTestData.summaryByDecile.map((row, idx) => ({
        bin: String(row.decile ?? row.bin ?? `D${idx + 1}`),
        meanPredicted: Number(row.meanPredicted ?? row.avgPred ?? row.predicted ?? 0),
        observedRate: Number(row.observedRate ?? row.actualRate ?? row.observed ?? 0),
        count: Number(row.count ?? row.n ?? 0),
      }));
    }
    if (testRecords.length === 0) return [];

    const sorted = [...testRecords].sort((a, b) => {
      const pa = typeof a.predictedProb === 'number' ? a.predictedProb : Number(a.prob ?? 0);
      const pb = typeof b.predictedProb === 'number' ? b.predictedProb : Number(b.prob ?? 0);
      return pa - pb;
    });
    const bins: CalibrationBin[] = [];
    const nBins = 10;
    for (let i = 0; i < nBins; i++) {
      const start = Math.floor((i * sorted.length) / nBins);
      const end = Math.floor(((i + 1) * sorted.length) / nBins);
      const slice = sorted.slice(start, end);
      if (slice.length === 0) continue;
      let sumP = 0;
      let sumY = 0;
      for (const r of slice) {
        sumP += typeof r.predictedProb === 'number' ? r.predictedProb : Number(r.prob ?? 0);
        const y =
          typeof r.actual === 'number'
            ? r.actual
            : typeof r.actualInformal === 'number'
            ? r.actualInformal
            : Number(r.actualFormal ?? 0);
        sumY += y;
      }
      bins.push({
        bin: `D${i + 1}`,
        meanPredicted: Number((sumP / slice.length).toFixed(4)),
        observedRate: Number((sumY / slice.length).toFixed(4)),
        count: slice.length,
      });
    }
    return bins;
  }, [testRecords]);

  const groupRows = useMemo<GroupSummaryRow[]>(() => {
    if (Array.isArray(trainSummary.byGroup) && trainSummary.byGroup.length > 0) {
      return trainSummary.byGroup;
    }
    const merged: GroupSummaryRow[] = [];
    if (Array.isArray(trainSummary.byEducation)) {
      for (const r of trainSummary.byEducation) merged.push({ ...r, group: r.group || 'Educación' });
    }
    if (Array.isArray(trainSummary.byDepartment)) {
      for (const r of trainSummary.byDepartment)
        merged.push({ ...r, group: r.group || 'Departamento' });
    }
    if (Array.isArray(trainSummary.byAgeGroup)) {
      for (const r of trainSummary.byAgeGroup) merged.push({ ...r, group: r.group || 'Edad' });
    }
    return merged;
  }, []);

  const filteredFeatures = useMemo(() => {
    const q = featureSearch.trim().toLowerCase();
    const list = q
      ? normalizedFeatures.filter(
          (f) =>
            f.name.toLowerCase().includes(q) ||
            (f.label && f.label.toLowerCase().includes(q)) ||
            (f.category && f.category.toLowerCase().includes(q))
        )
      : [...normalizedFeatures];

    if (sortByAbsCoef) {
      list.sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
    }
    return list;
  }, [normalizedFeatures, featureSearch, sortByAbsCoef]);

  const topChartFeatures = useMemo(() => {
    return [...normalizedFeatures]
      .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
      .slice(0, 15)
      .map((f) => ({
        name: (f.label || f.name).slice(0, 28),
        fullName: f.label || f.name,
        coef: Number(f.coef.toFixed(4)),
        oddsRatio: Number((f.oddsRatio ?? Math.exp(f.coef)).toFixed(4)),
      }));
  }, [normalizedFeatures]);

  const simulationResult = useMemo(() => {
    let linearPredictor = interceptValue;
    const contributions: Array<{ name: string; label: string; value: number; contribution: number }> =
      [];

    for (const f of normalizedFeatures) {
      const val = typeof simInputs[f.name] === 'number' ? simInputs[f.name] : 0;
      const contrib = f.coef * val;
      linearPredictor += contrib;
      contributions.push({
        name: f.name,
        label: f.label || f.name,
        value: val,
        contribution: contrib,
      });
    }

    const probability = sigmoid(linearPredictor);
    const predictedClass = probability >= threshold ? 1 : 0;
    contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    return {
      linearPredictor,
      probability,
      predictedClass,
      contributions,
    };
  }, [interceptValue, normalizedFeatures, simInputs, threshold]);

  const handleResetSimulator = () => {
    const reset: Record<string, number> = {};
    for (const f of normalizedFeatures) {
      reset[f.name] = typeof f.mean === 'number' ? f.mean : 0;
    }
    setSimInputs(reset);
  };

  const totalTrainRows = trainSummary.nRows ?? modelParams.metrics?.nSamples;
  const totalTestRows = lockedTestData.nRows ?? (testRecords.length > 0 ? testRecords.length : lockedTestData.metrics?.nSamples);
  const aucScore =
    activeThresholdMetrics.auc ??
    lockedTestData.metrics?.auc ??
    lockedTestData.metrics?.rocAuc ??
    modelParams.metrics?.auc;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <header className="bg-slate-900 text-white border-b border-slate-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-widest text-sky-400 font-semibold">
                Uruguay · Encuesta Continua de Hogares (ECH 2025)
              </p>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight mt-1">
                {modelParams.modelName || 'Modelo Predictivo de Formalidad / Informalidad Laboral'}
              </h1>
              <p className="text-sm text-slate-300 mt-1">
                Regresión logística evaluada en conjunto de prueba bloqueado (Locked Test Set)
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-xs">
              <div className="bg-slate-800/90 border border-slate-700 rounded-lg px-3.5 py-2">
                <span className="text-slate-400 block">Muestra Entrenamiento</span>
                <span className="text-base font-semibold text-white">
                  {formatInt(totalTrainRows)}
                </span>
              </div>
              <div className="bg-slate-800/90 border border-slate-700 rounded-lg px-3.5 py-2">
                <span className="text-slate-400 block">Locked Test Set</span>
                <span className="text-base font-semibold text-white">
                  {formatInt(totalTestRows)}
                </span>
              </div>
              <div className="bg-slate-800/90 border border-slate-700 rounded-lg px-3.5 py-2">
                <span className="text-slate-400 block">ROC-AUC (Test)</span>
                <span className="text-base font-semibold text-emerald-400">
                  {formatNum(aucScore, 4)}
                </span>
              </div>
            </div>
          </div>

          <nav className="flex flex-wrap gap-2 mt-6 pt-4 border-t border-slate-800">
            {[
              { id: 'overview', label: '1. Resumen y Datos ECH' },
              { id: 'coefficients', label: '2. Coeficientes y Odds Ratios' },
              { id: 'evaluation', label: '3. Evaluación Locked Test' },
              { id: 'simulator', label: '4. Simulador Interactivo' },
              { id: 'methodology', label: '5. Formulación Matemática' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id as TabKey)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-sky-500 text-white shadow-sm'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {activeTab === 'overview' && (
          <section className="space-y-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Observaciones Train
                </p>
                <p className="text-2xl font-bold text-slate-900 mt-1">
                  {formatInt(totalTrainRows)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Variables activas: {normalizedFeatures.length}
                </p>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Tasa Base (Train)
                </p>
                <p className="text-2xl font-bold text-sky-700 mt-1">
                  {formatPct(
                    trainSummary.informalityRate ??
                      trainSummary.formalityRate ??
                      trainSummary.metrics?.positiveRate
                  )}
                </p>
                <p className="text-xs text-slate-500 mt-1">Prevalencia de clase positiva</p>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  ROC-AUC (Locked Test)
                </p>
                <p className="text-2xl font-bold text-emerald-700 mt-1">
                  {formatNum(aucScore, 4)}
                </p>
                <p className="text-xs text-slate-500 mt-1">Capacidad discriminativa fuera de muestra</p>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                  Accuracy (Umbral {threshold.toFixed(2)})
                </p>
                <p className="text-2xl font-bold text-slate-900 mt-1">
                  {formatPct(activeThresholdMetrics.accuracy)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  F1-Score: {formatNum(activeThresholdMetrics.f1, 3)}
                </p>
              </div>
            </div>

            {groupRows.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900">
                  Distribución por Subgrupos en Entrenamiento (ECH 2025)
                </h2>
                <p className="text-sm text-slate-600 mt-1 mb-4">
                  Comparación de tasas observadas y probabilidades medias predichas por segmento.
                </p>
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={groupRows.slice(0, 14).map((r) => ({
                        name: String(r.subGroup || r.group),
                        observada: Number(
                          ((r.informalityRate ?? r.formalityRate ?? 0) * 100).toFixed(2)
                        ),
                        predicha: Number(((r.avgPredictedProb ?? 0) * 100).toFixed(2)),
                      }))}
                      margin={{ top: 10, right: 20, left: 0, bottom: 40 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} height={60} tick={{ fontSize: 11 }} />
                      <YAxis unit="%" tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="observada" name="Tasa Observada (%)" fill="#0284c7" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="predicha" name="Prob. Media Predicha (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-900">
                Principales Predictores del Modelo (Top 15 por magnitud de <InlineMath math="|\beta_j|" />)
              </h2>
              <p className="text-sm text-slate-600 mt-1 mb-4">
                Coeficientes log-odds estimados en <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">ech_model_params.json</code>.
              </p>
              <div className="h-96 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={topChartFeatures}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis type="number" tick={{ fontSize: 12 }} />
                    <YAxis type="category" dataKey="name" width={135} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <ReferenceLine x={0} stroke="#64748b" />
                    <Bar dataKey="coef" name="Coeficiente (β)" fill="#0f172a" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'coefficients' && (
          <section className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Tabla Completa de Parámetros (<InlineMath math="\beta_0 =" /> {formatNum(interceptValue, 4)})
                </h2>
                <p className="text-sm text-slate-600">
                  Coeficientes log-odds y razones de momios (<InlineMath math="OR = e^{\beta_j}" />).
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <input
                  type="text"
                  value={featureSearch}
                  onChange={(e) => setFeatureSearch(e.target.value)}
                  placeholder="Filtrar variable..."
                  className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500"
                />
                <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={sortByAbsCoef}
                    onChange={(e) => setSortByAbsCoef(e.target.checked)}
                    className="rounded border-slate-300 text-sky-600 focus:ring-sky-500"
                  />
                  Ordenar por magnitud
                </label>
              </div>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded-lg">
              <table className="min-w-full divide-y divide-slate-200 text-sm">
                <thead className="bg-slate-100 text-slate-700">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold">Variable</th>
                    <th className="px-4 py-3 text-right font-semibold">Coeficiente (β)</th>
                    <th className="px-4 py-3 text-right font-semibold">Odds Ratio (e^β)</th>
                    <th className="px-4 py-3 text-right font-semibold">Error Estándar</th>
                    <th className="px-4 py-3 text-right font-semibold">p-valor</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  <tr className="bg-slate-50/70 font-medium">
                    <td className="px-4 py-2.5 text-slate-800">(Intercepto)</td>
                    <td className="px-4 py-2.5 text-right font-mono">{formatNum(interceptValue, 4)}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{formatNum(Math.exp(interceptValue), 4)}</td>
                    <td className="px-4 py-2.5 text-right font-mono">—</td>
                    <td className="px-4 py-2.5 text-right font-mono">—</td>
                  </tr>
                  {filteredFeatures.map((feat) => (
                    <tr key={feat.name} className="hover:bg-slate-50">
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-slate-900">{feat.label || feat.name}</div>
                        {feat.label && feat.label !== feat.name && (
                          <div className="text-xs text-slate-400 font-mono">{feat.name}</div>
                        )}
                      </td>
                      <td
                        className={`px-4 py-2.5 text-right font-mono font-semibold ${
                          feat.coef >= 0 ? 'text-sky-700' : 'text-amber-700'
                        }`}
                      >
                        {feat.coef >= 0 ? `+${feat.coef.toFixed(4)}` : feat.coef.toFixed(4)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono">
                        {formatNum(feat.oddsRatio ?? Math.exp(feat.coef), 4)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-600">
                        {formatNum(feat.stdError, 4)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-600">
                        {typeof feat.pValue === 'number'
                          ? feat.pValue < 0.001
                            ? '< 0.001'
                            : feat.pValue.toFixed(4)
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {activeTab === 'evaluation' && (
          <section className="space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    Sensibilidad al Umbral de Decisión (<InlineMath math="\tau" />)
                  </h2>
                  <p className="text-sm text-slate-600">
                    Ajusta el punto de corte <InlineMath math="\hat{p}_i \ge \tau" /> sobre el conjunto de prueba bloqueado.
                  </p>
                </div>
                <div className="flex items-center gap-4 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5">
                  <label htmlFor="threshold-slider" className="text-sm font-medium text-slate-700">
                    Umbral <InlineMath math="\tau" />:
                  </label>
                  <input
                    id="threshold-slider"
                    type="range"
                    min={0.05}
                    max={0.95}
                    step={0.01}
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                    className="w-40 accent-sky-600"
                  />
                  <span className="font-mono font-bold text-sky-700 w-12 text-right">
                    {threshold.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 mt-6">
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs text-slate-500 block">Accuracy</span>
                  <span className="text-xl font-bold text-slate-900">
                    {formatPct(activeThresholdMetrics.accuracy)}
                  </span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs text-slate-500 block">Precision</span>
                  <span className="text-xl font-bold text-slate-900">
                    {formatPct(activeThresholdMetrics.precision)}
                  </span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs text-slate-500 block">Recall (TPR)</span>
                  <span className="text-xl font-bold text-slate-900">
                    {formatPct(activeThresholdMetrics.recall)}
                  </span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs text-slate-500 block">F1-Score</span>
                  <span className="text-xl font-bold text-sky-700">
                    {formatNum(activeThresholdMetrics.f1, 3)}
                  </span>
                </div>
                <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs text-slate-500 block">ROC-AUC</span>
                  <span className="text-xl font-bold text-emerald-700">
                    {formatNum(activeThresholdMetrics.auc, 4)}
                  </span>
                </div>
              </div>

              {typeof activeThresholdMetrics.tp === 'number' && (
                <div className="mt-6 max-w-md">
                  <h3 className="text-sm font-semibold text-slate-800 mb-2">
                    Matriz de Confusión (Locked Test Set)
                  </h3>
                  <table className="w-full text-sm border border-slate-200 rounded-lg overflow-hidden">
                    <thead className="bg-slate-100">
                      <tr>
                        <th className="p-2.5 text-left border-b border-slate-200">Real \ Predicho</th>
                        <th className="p-2.5 text-right border-b border-slate-200">Pred = 1</th>
                        <th className="p-2.5 text-right border-b border-slate-200">Pred = 0</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="p-2.5 font-medium border-b border-slate-200">Real = 1</td>
                        <td className="p-2.5 text-right font-mono bg-emerald-50 text-emerald-900 font-semibold border-b border-slate-200">
                          TP: {formatInt(activeThresholdMetrics.tp)}
                        </td>
                        <td className="p-2.5 text-right font-mono bg-rose-50 text-rose-900 border-b border-slate-200">
                          FN: {formatInt(activeThresholdMetrics.fn)}
                        </td>
                      </tr>
                      <tr>
                        <td className="p-2.5 font-medium">Real = 0</td>
                        <td className="p-2.5 text-right font-mono bg-amber-50 text-amber-900">
                          FP: {formatInt(activeThresholdMetrics.fp)}
                        </td>
                        <td className="p-2.5 text-right font-mono bg-sky-50 text-sky-900 font-semibold">
                          TN: {formatInt(activeThresholdMetrics.tn)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {rocData.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                  <h3 className="text-base font-semibold text-slate-900">
                    Curva ROC (AUC = {formatNum(activeThresholdMetrics.auc, 4)})
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 mb-4">
                    Tasa de Verdaderos Positivos (TPR) vs. Falsos Positivos (FPR)
                  </p>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={rocData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis
                          dataKey="fpr"
                          type="number"
                          domain={[0, 1]}
                          tick={{ fontSize: 11 }}
                          label={{ value: 'FPR', position: 'insideBottom', offset: -5, fontSize: 12 }}
                        />
                        <YAxis
                          domain={[0, 1]}
                          tick={{ fontSize: 11 }}
                          label={{ value: 'TPR', angle: -90, position: 'insideLeft', fontSize: 12 }}
                        />
                        <Tooltip />
                        <Area
                          type="monotone"
                          dataKey="tpr"
                          name="TPR (Sensibilidad)"
                          stroke="#0284c7"
                          fill="#bae6fd"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {calibrationData.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                  <h3 className="text-base font-semibold text-slate-900">
                    Calibración por Deciles de Riesgo
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 mb-4">
                    Probabilidad media predicha frente a frecuencia observada en Locked Test
                  </p>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={calibrationData}
                        margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="meanPredicted"
                          name="Prob. Predicha"
                          stroke="#0284c7"
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="observedRate"
                          name="Tasa Observada"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>

            {thresholdCurve.length > 0 && (
              <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h3 className="text-base font-semibold text-slate-900">
                  Métricas de Clasificación por Umbral
                </h3>
                <div className="h-72 w-full mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="threshold" name="Umbral" domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Legend />
                      <ReferenceLine x={threshold} stroke="#ef4444" strokeDasharray="4 4" />
                      <Scatter name="F1-Score" data={thresholdCurve} dataKey="f1" fill="#0284c7" line />
                      <Scatter name="Precision" data={thresholdCurve} dataKey="precision" fill="#10b981" line />
                      <Scatter name="Recall" data={thresholdCurve} dataKey="recall" fill="#f59e0b" line />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </section>
        )}

        {activeTab === 'simulator' && (
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    Simulador de Perfil Individual
                  </h2>
                  <p className="text-sm text-slate-600">
                    Modifica los valores de las covariables <InlineMath math="x_j" /> para evaluar la probabilidad predicha.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleResetSimulator}
                  className="px-3 py-1.5 text-xs font-medium bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition-colors"
                >
                  Restablecer valores
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[540px] overflow-y-auto pr-1">
                {normalizedFeatures.map((feat) => {
                  const val = simInputs[feat.name] ?? 0;
                  const isBinary =
                    feat.type === 'binary' ||
                    (feat.min === 0 && feat.max === 1) ||
                    (feat.min === undefined && feat.max === undefined && (val === 0 || val === 1));

                  return (
                    <div
                      key={feat.name}
                      className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/60 flex flex-col justify-between"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <label
                          htmlFor={`sim-${feat.name}`}
                          className="text-xs font-semibold text-slate-800 leading-snug"
                        >
                          {feat.label || feat.name}
                        </label>
                        <span className="text-[11px] font-mono text-slate-500 shrink-0">
                          β={feat.coef >= 0 ? `+${feat.coef.toFixed(2)}` : feat.coef.toFixed(2)}
                        </span>
                      </div>

                      {isBinary ? (
                        <div className="mt-2.5 flex gap-2">
                          {[0, 1].map((opt) => (
                            <button
                              key={opt}
                              id={opt === 1 ? `sim-${feat.name}` : undefined}
                              type="button"
                              onClick={() =>
                                setSimInputs((prev) => ({ ...prev, [feat.name]: opt }))
                              }
                              className={`flex-1 py-1 text-xs font-medium rounded border transition-colors ${
                                val === opt
                                  ? 'bg-sky-600 text-white border-sky-600'
                                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                              }`}
                            >
                              {opt === 1 ? 'Sí (1)' : 'No (0)'}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <input
                          id={`sim-${feat.name}`}
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => {
                            const parsed = parseFloat(e.target.value);
                            setSimInputs((prev) => ({
                              ...prev,
                              [feat.name]: Number.isNaN(parsed) ? 0 : parsed,
                            }));
                          }}
                          className="mt-2 w-full border border-slate-300 rounded px-2.5 py-1 text-sm font-mono bg-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5 h-fit">
              <h3 className="text-base font-semibold text-slate-900">Resultado de la Predicción</h3>

              <div className="p-5 rounded-xl bg-slate-900 text-white text-center">
                <span className="text-xs uppercase tracking-wider text-slate-400 block">
                  Probabilidad Estimada <InlineMath math="\hat{p}" />
                </span>
                <span className="text-4xl font-bold text-sky-400 block mt-1">
                  {(simulationResult.probability * 100).toFixed(2)}%
                </span>
                <span className="text-xs text-slate-300 block mt-2">
                  Logit <InlineMath math="z =" /> {simulationResult.linearPredictor.toFixed(4)}
                </span>
                <div className="mt-3">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                      simulationResult.predictedClass === 1
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-400/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/40'
                    }`}
                  >
                    Clasificación (τ={threshold.toFixed(2)}):{' '}
                    {simulationResult.predictedClass === 1 ? 'Clase Positiva (1)' : 'Clase Base (0)'}
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                  Mayores Contribuciones Activas (<InlineMath math="\beta_j x_j" />)
                </h4>
                <ul className="space-y-2 text-xs">
                  {simulationResult.contributions
                    .filter((c) => c.contribution !== 0)
                    .slice(0, 6)
                    .map((c) => (
                      <li
                        key={c.name}
                        className="flex items-center justify-between border-b border-slate-100 pb-1.5"
                      >
                        <span className="text-slate-700 truncate pr-2">{c.label}</span>
                        <span
                          className={`font-mono font-semibold shrink-0 ${
                            c.contribution >= 0 ? 'text-sky-700' : 'text-amber-700'
                          }`}
                        >
                          {c.contribution >= 0
                            ? `+${c.contribution.toFixed(3)}`
                            : c.contribution.toFixed(3)}
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'methodology' && (
          <section className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5 text-slate-800">
            <h2 className="text-lg font-semibold text-slate-900">
              Especificación Econométrica y Protocolo de Evaluación
            </h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              La probabilidad condicional de pertenecer a la categoría objetivo en la Encuesta Continua de Hogares (ECH 2025) se modela mediante regresión logística binaria:
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 overflow-x-auto">
              <BlockMath math="P(Y_i = 1 \mid \mathbf{x}_i) = \sigma(z_i) = rac{1}{1 + \exp\left(-\left(eta_0 + \sum_{j=1}^{p} eta_j x_{ij}ight)ight)}" />
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">
              La clasificación binaria aplica una regla de umbral <InlineMath math="\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)" /> evaluada estrictamente sobre el conjunto de prueba bloqueado (<code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">ech_locked_test.json</code>), evitando fuga de información entre estimación y validación.
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 overflow-x-auto">
              <BlockMath math="	ext{OR}_j = \exp(eta_j), \qquad \mathcal{L}(oldsymbol{eta}) = -rac{1}{N}\sum_{i=1}^{N}\left[y_i \ln(\hat{p}_i) + (1 - y_i)\ln(1 - \hat{p}_i)ight]" />
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;

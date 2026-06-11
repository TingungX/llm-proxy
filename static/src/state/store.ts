export { configSignal, modelsSignal, errorHandlingSignal, setConfig, getMappingName } from './config';
export { endpointsSignal, endpointNameMapSignal, setEndpoints, getEndpoint } from './endpoints';
export {
  usageModeSignal, usageGroupBySignal, hourlyDayOffsetSignal,
  heatmapDaysSignal, usageEndpointFilterSignal, heatmapEndpointFilterSignal,
  usageChartRef, endpointUsageChartRef,
  usageRefreshTrigger, heatmapRefreshTrigger,
  setHeatmapDays, setUsageMode, setUsageGroupBy, setHourlyDayOffset,
  setUsageEndpointFilter, setHeatmapEndpointFilter,
  loadHeatmapDaysFromLocalStorage,
  HEATMAP_DAYS_OPTIONS,
} from './usage';
export type { UsageMode, UsageGroupBy } from './usage';
export {
  logsSignal, logsTotalSignal, logsOffsetSignal, logsLimitSignal,
  logsLoadingSignal, logFilterSignal, hasActiveFiltersSignal,
  mergeConsecutiveSignal, setMergeConsecutive, loadMergeConsecutiveFromLocalStorage,
  filterDrawerOpenSignal, activeFilterCountSignal,
} from './logs';
export {
  currentEndpointIdSignal, epModelFilterSignal, epModelSortSignal,
  selectEndpoint, setEpModelFilter, setEpModelSort,
} from './endpointPage';
export type { EpModelSort } from './endpointPage';
export {
  modelModalOpen, modelModalEditing, openModelModal, closeModelModal,
  endpointModalOpen, endpointModalEditing, openEndpointModal, closeEndpointModal,
  mappingModalOpen, mappingModalEndpointId, openMappingModal, closeMappingModal,
} from './modals';

import { setConfig } from './config';
import { setEndpoints } from './endpoints';
import { fetchConfig } from '../api/config';
import { fetchEndpoints } from '../api/endpoints';

export async function initStore(): Promise<void> {
  const [config, endpoints] = await Promise.all([
    fetchConfig(),
    fetchEndpoints(),
  ]);
  setConfig(config);
  setEndpoints(endpoints);
}

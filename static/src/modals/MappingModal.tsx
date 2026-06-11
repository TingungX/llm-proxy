import { useEffect, useState } from 'preact/hooks';
import {
  mappingModalOpen, mappingModalEndpointId, closeMappingModal,
  modelsSignal, endpointsSignal,
} from '../state/store';
import { showToast } from '../components/Toast';
import { ApiCallError } from '../api/client';
import { esc } from '../utils/format';
import { Modal } from '../components/Modal';
import { EmptyState } from '../components/EmptyState';
import { saveEndpoint, fetchEndpoints } from '../api/endpoints';
interface Route {
  family: string;
  model: string;
}

export function MappingModal() {
  const isOpen = mappingModalOpen.value;
  const epId = mappingModalEndpointId.value;
  const [routes, setRoutes] = useState<Route[]>([]);

  useEffect(() => {
    if (!isOpen || !epId) return;
    const ep = endpointsSignal.value.find(e => e.endpoint_id === epId);
    const initialRoutes: Route[] = ep
      ? Object.entries(ep.family_routing ?? {}).map(([family, model]) => ({ family, model }))
      : [];
    setRoutes(initialRoutes);
  }, [isOpen, epId]);

  if (!isOpen) return null;

  const models = Object.keys(modelsSignal.value);

  const addRoute = () => setRoutes([...routes, { family: '', model: models[0] ?? '' }]);
  const removeRoute = (i: number) => setRoutes(routes.filter((_, idx) => idx !== i));
  const updateRoute = (i: number, field: keyof Route, value: string) => {
    setRoutes(routes.map((r, idx) => idx === i ? { ...r, [field]: value } : r));
  };

  const onSave = async () => {
    if (!epId) return;
    const family_routing: Record<string, string> = {};
    for (const r of routes) {
      if (r.family && r.model) family_routing[r.family] = r.model;
    }
    try {
      await saveEndpoint(epId, { family_routing });
      const eps = await fetchEndpoints();
      endpointsSignal.value = eps;
      showToast('路由已保存', 'ok');
      closeMappingModal();
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '保存失败', 'err');
    }
  };

  return (
    <Modal onClose={closeMappingModal} size="sm">
      <h2 class="modal-title">路由配置</h2>
      {routes.length === 0 ? (
        <EmptyState>暂无路由。点击"添加"创建。</EmptyState>
      ) : (
        <table style="margin-top: 16px;">
          <thead><tr><th>客户端名称</th><th>实际模型</th><th></th></tr></thead>
          <tbody>
            {routes.map((r, i) => (
              <tr key={i}>
                <td><input value={r.family} onInput={(e: Event) => updateRoute(i, 'family', (e.target as HTMLInputElement).value)} placeholder="opus-4-7" /></td>
                <td>
                  <select value={r.model} onChange={(e: Event) => updateRoute(i, 'model', (e.target as HTMLSelectElement).value)}>
                    {models.map(m => <option value={m}>{esc(m)}</option>)}
                  </select>
                </td>
                <td><button class="icon" onClick={() => removeRoute(i)}>×</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div style="margin-top: 8px;">
        <button onClick={addRoute}>+ 添加</button>
      </div>
      <div class="modal-actions">
        <button onClick={closeMappingModal}>取消</button>
        <button class="primary" onClick={onSave}>保存</button>
      </div>
    </Modal>
  );
}

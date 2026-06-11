import type { JSX } from 'preact';
import type { Endpoint } from '../api/types';
import { EDIT_ICON, DELETE_ICON } from '../utils/icons';
import { openEndpointModal } from '../state/store';
import { saveEndpoint, deleteEndpoint } from '../api/endpoints';
import { fetchEndpoints } from '../api/endpoints';
import { setEndpoints } from '../state/store';
import { showToast } from './Toast';
import { ApiCallError } from '../api/client';

export interface EndpointDockProps {
  endpoint: Endpoint;
}

export function EndpointDock({ endpoint }: EndpointDockProps): JSX.Element {
  const handleToggle = async () => {
    try {
      await saveEndpoint(endpoint.endpoint_id, { enabled: !endpoint.enabled });
      const list = await fetchEndpoints();
      setEndpoints(list);
      showToast(endpoint.enabled ? '已停用' : '已启用', 'ok');
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '更新失败', 'err');
    }
  };

  const handleDelete = async () => {
    if (endpoint.endpoint_id === 'default') return;
    if (!confirm(`删除端点 ${endpoint.name}？`)) return;
    try {
      await deleteEndpoint(endpoint.endpoint_id);
      const list = await fetchEndpoints();
      setEndpoints(list);
      showToast('端点已删除', 'ok');
    } catch (e) {
      showToast(e instanceof ApiCallError ? e.detail : '删除失败', 'err');
    }
  };

  const isDefault = endpoint.is_default === true;

  return (
    <div class="endpoint-dock">
      <button
        type="button"
        class={`protocol-chip${endpoint.enabled ? ' active' : ''}`}
        onClick={handleToggle}
      >
        ● 启用
      </button>
      <button
        type="button"
        class="icon"
        onClick={() => openEndpointModal(endpoint.endpoint_id)}
        title="编辑"
        dangerouslySetInnerHTML={{ __html: EDIT_ICON }}
      />
      <button
        type="button"
        class="icon icon-danger"
        onClick={handleDelete}
        disabled={isDefault}
        title="删除"
        dangerouslySetInnerHTML={{ __html: DELETE_ICON }}
      />
    </div>
  );
}

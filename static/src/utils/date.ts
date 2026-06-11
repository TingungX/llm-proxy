export function localDateTimeToBackend(localDateTime: string): string {
  if (!localDateTime) return '';
  const d = new Date(localDateTime);
  if (isNaN(d.getTime())) return '';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function localDateToBackendPrefix(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function toLocalDateString(beijing: string): string {
  if (/^\d{4}-\d{2}-\d{2}$/.test(beijing)) {
    return beijing;
  }
  const m = /^(\d{4}-\d{2}-\d{2}) /.exec(beijing);
  return m ? m[1] : beijing.slice(0, 10);
}

export function formatDateTime(beijing: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/.exec(beijing);
  if (!m) return beijing;
  return `${m[2]}/${m[3]} ${m[4]}:${m[5]}`;
}

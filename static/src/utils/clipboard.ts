export async function copyToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    throw new Error('Failed to copy to clipboard');
  }
}

export function togglePasswordVisibility(input: HTMLInputElement): boolean {
  const isPassword = input.type === 'password';
  input.type = isPassword ? 'text' : 'password';
  return isPassword;
}

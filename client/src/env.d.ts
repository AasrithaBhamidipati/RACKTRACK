/// <reference types="vite/client" />

// Vite-specific import.meta helpers used in the project (glob, globEager)
// Declaring them here silences TypeScript errors about missing properties on ImportMeta.
interface ImportMeta {
  /**
   * Returns a record of file paths to async import functions.
   * Example: const modules = import.meta.glob('./dir/*.ts')
   */
  glob(pattern: string): Record<string, () => Promise<{ default: any }>>;

  /**
   * Eagerly imports matched modules and returns an object mapping paths to module objects.
   * Example: const modules = import.meta.globEager('./dir/*.png')
   */
  globEager(pattern: string): Record<string, { default: any }>;
}

export {};

export function openDB() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open("racktrack-db", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("analyzing_files")) {
        db.createObjectStore("analyzing_files", { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveAnalyzingFiles(filesData: any): Promise<string> {
  const db = await openDB();
  return new Promise<string>((resolve, reject) => {
    const tx = db.transaction("analyzing_files", "readwrite");
    const store = tx.objectStore("analyzing_files");
    const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
    const record = { id, filesData, createdAt: Date.now() };
    const req = store.add(record);
    req.onsuccess = () => resolve(id);
    req.onerror = () => reject(req.error);
  });
}

export async function getAnalyzingFilesById(id: string): Promise<any | null> {
  const db = await openDB();
  return new Promise<any | null>((resolve, reject) => {
    const tx = db.transaction("analyzing_files", "readonly");
    const store = tx.objectStore("analyzing_files");
    const req = store.get(id);
    req.onsuccess = () => {
      resolve(req.result ? req.result.filesData : null);
    };
    req.onerror = () => reject(req.error);
  });
}

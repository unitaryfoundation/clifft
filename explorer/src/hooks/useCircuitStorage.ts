import { useState, useCallback } from "react";

export interface SavedCircuit {
  id: string;
  name: string;
  timestamp: number;
  source: string;
}

const DRAFT_KEY = "ucc-current-draft";
const SAVED_KEY = "ucc-saved-circuits";
const MAX_SAVED = 15;

function loadSaved(): SavedCircuit[] {
  try {
    const raw = localStorage.getItem(SAVED_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as SavedCircuit[];
  } catch {
    return [];
  }
}

export function saveDraft(source: string): void {
  try {
    localStorage.setItem(DRAFT_KEY, source);
  } catch {
    // quota exceeded — ignore
  }
}

export function loadDraft(): string | null {
  return localStorage.getItem(DRAFT_KEY);
}

export function useCircuitStorage() {
  const [saved, setSaved] = useState<SavedCircuit[]>(loadSaved);

  const saveCircuit = useCallback((name: string, source: string) => {
    setSaved((prev) => {
      const entry: SavedCircuit = {
        id: crypto.randomUUID(),
        name,
        timestamp: Date.now(),
        source,
      };
      const next = [entry, ...prev].slice(0, MAX_SAVED);
      try {
        localStorage.setItem(SAVED_KEY, JSON.stringify(next));
      } catch {
        // quota exceeded
      }
      return next;
    });
  }, []);

  const deleteCircuit = useCallback((id: string) => {
    setSaved((prev) => {
      const next = prev.filter((c) => c.id !== id);
      try {
        localStorage.setItem(SAVED_KEY, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  return { saved, saveCircuit, deleteCircuit };
}

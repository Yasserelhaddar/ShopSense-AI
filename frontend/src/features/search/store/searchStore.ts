import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { SearchFilters, SortOption, ViewMode } from '@/types/common.types'

interface SearchState {
  filters: SearchFilters
  sortBy: SortOption
  viewMode: ViewMode
  updateFilters: (filters: Partial<SearchFilters>) => void
  setSortBy: (sort: SortOption) => void
  setViewMode: (mode: ViewMode) => void
  clearFilters: () => void
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set) => ({
      filters: {},
      sortBy: 'relevance',
      viewMode: 'grid',
      updateFilters: (newFilters) =>
        set((state) => ({
          filters: { ...state.filters, ...newFilters },
        })),
      setSortBy: (sortBy) => set({ sortBy }),
      setViewMode: (viewMode) => set({ viewMode }),
      clearFilters: () => set({ filters: {} }),
    }),
    {
      name: 'search-storage',
    }
  )
)

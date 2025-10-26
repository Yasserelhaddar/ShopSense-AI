import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ComparisonState {
  selectedProductIds: string[]
  addProduct: (id: string) => void
  removeProduct: (id: string) => void
  clearProducts: () => void
  isSelected: (id: string) => boolean
}

export const useComparisonStore = create<ComparisonState>()(
  persist(
    (set, get) => ({
      selectedProductIds: [],
      addProduct: (id) =>
        set((state) => {
          if (state.selectedProductIds.includes(id)) return state
          if (state.selectedProductIds.length >= 4) {
            // Max 4 products for comparison
            return state
          }
          return { selectedProductIds: [...state.selectedProductIds, id] }
        }),
      removeProduct: (id) =>
        set((state) => ({
          selectedProductIds: state.selectedProductIds.filter((pid) => pid !== id),
        })),
      clearProducts: () => set({ selectedProductIds: [] }),
      isSelected: (id) => get().selectedProductIds.includes(id),
    }),
    {
      name: 'comparison-storage',
    }
  )
)

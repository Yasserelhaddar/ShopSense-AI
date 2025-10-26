import { useMutation } from '@tanstack/react-query'
import { compareProducts, type ComparisonRequest } from '../api/productsApi'

export function useComparison() {
  return useMutation({
    mutationFn: (request: ComparisonRequest) => compareProducts(request),
  })
}

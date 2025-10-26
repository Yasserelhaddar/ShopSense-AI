import { useQuery } from '@tanstack/react-query'
import { searchProducts, type SearchRequest } from '../api/searchApi'

export function useSearch(request: SearchRequest, enabled = true) {
  return useQuery({
    queryKey: ['search', request.query, request.budget_range, request.filters, request.limit],
    queryFn: () => searchProducts(request),
    enabled: enabled && request.query.length >= 2,
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

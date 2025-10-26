import { useQuery } from '@tanstack/react-query'
import { getProductById } from '../api/productsApi'

export function useProduct(id: string) {
  return useQuery({
    queryKey: ['product', id],
    queryFn: () => getProductById(id),
    enabled: !!id,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

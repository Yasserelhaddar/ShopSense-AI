import { useQuery } from '@tanstack/react-query'
import { getTrending } from '../api/recommendationsApi'

export function useTrending(limit = 10) {
  return useQuery({
    queryKey: ['recommendations', 'trending', limit],
    queryFn: () => getTrending(limit),
    staleTime: 5 * 60 * 1000, // 5 minutes - trending can change
  })
}

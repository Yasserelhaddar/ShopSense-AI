import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getPreferences, updatePreferences, clearHistory, type UserPreferences } from '../api/userApi'

export function useUserPreferences() {
  return useQuery({
    queryKey: ['user', 'preferences'],
    queryFn: getPreferences,
    staleTime: Infinity, // Preferences don't change often
  })
}

export function useUpdatePreferences() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (preferences: UserPreferences) => updatePreferences(preferences),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user', 'preferences'] })
    },
  })
}

export function useClearHistory() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: clearHistory,
    onSuccess: () => {
      // Invalidate relevant queries after clearing history
      queryClient.invalidateQueries({ queryKey: ['user'] })
    },
  })
}

import { advisoryApi } from '@/lib/api/axios'
import type { UserPreferences } from '@/types/common.types'

export type { UserPreferences }

export interface FeedbackRequest {
  product_id: string
  rating?: number
  comment?: string
}

export const submitFeedback = async (feedback: FeedbackRequest): Promise<void> => {
  await advisoryApi.post('/feedback', feedback)
}

export const getPreferences = async (): Promise<UserPreferences> => {
  const { data } = await advisoryApi.get<UserPreferences>('/user/preferences')
  return data
}

export const updatePreferences = async (preferences: UserPreferences): Promise<UserPreferences> => {
  const { data} = await advisoryApi.put<UserPreferences>('/user/preferences', preferences)
  return data
}

export const clearHistory = async (): Promise<void> => {
  await advisoryApi.delete('/user/clear-history')
}

import { advisoryApi } from '@/lib/api/axios'
import type { Product, PriceRange } from '@/types/common.types'

export interface ConversationMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface AdviceRequest {
  query: string
  budget_range?: PriceRange
  context?: string[]
}

export interface BackendAdviceRequest {
  conversation_history: ConversationMessage[]
  user_context?: Record<string, any>
  specific_questions?: string[]
  consultation_type?: string
  model_id?: string
}

export interface ProductRecommendation {
  product_id: string
  title: string
  price: number
  store: string
  rating?: number
  image_url?: string
  product_url?: string
  match_score: number
  why_recommended: string
  key_benefits: string[]
}

export interface AdviceResponse {
  advice: string
  product_recommendations: ProductRecommendation[]
  next_steps: string[]
  confidence_score: number
  reasoning: string
}

export const getAdvice = async (request: AdviceRequest): Promise<AdviceResponse> => {
  // Transform frontend request to backend format
  const backendRequest: BackendAdviceRequest = {
    conversation_history: [
      {
        role: 'user',
        content: request.query,
      },
    ],
    user_context: request.budget_range
      ? { budget_range: request.budget_range }
      : undefined,
    specific_questions: request.context,
    consultation_type: 'general',
  }

  const { data } = await advisoryApi.post<AdviceResponse>('/advice', backendRequest)
  return data
}

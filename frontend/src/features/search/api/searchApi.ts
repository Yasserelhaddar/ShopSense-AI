import { advisoryApi } from '@/lib/api/axios'
import type { Product, SearchFilters, PriceRange } from '@/types/common.types'

export interface SearchRequest {
  query: string
  budget_range?: PriceRange
  filters?: SearchFilters
  limit?: number
}

export interface SearchResponse {
  products: Product[]
  total: number
  query: string
}

export interface BackendSearchProduct {
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

export interface BackendSearchResponse {
  search_results: BackendSearchProduct[]
  ai_advice: string
  follow_up_questions: string[]
  search_insights: Record<string, any>
  total_results: number
  processing_time_ms: number
}

export const searchProducts = async (request: SearchRequest): Promise<SearchResponse> => {
  const { data } = await advisoryApi.post<BackendSearchResponse>('/search', request)

  // Transform backend format to frontend format
  const products: Product[] = (data.search_results || []).map((item) => ({
    id: item.product_id,
    title: item.title,
    price: item.price,
    store: item.store,
    rating: item.rating,
    image: item.image_url || '/placeholder-image.jpg',
    url: item.product_url,
    description: item.why_recommended,
  }))

  return {
    products,
    total: data.total_results || 0,
    query: request.query,
  }
}

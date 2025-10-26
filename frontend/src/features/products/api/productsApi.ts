import { advisoryApi, discoveryApi } from '@/lib/api/axios'
import type { Product } from '@/types/common.types'

export interface ComparisonCriteria {
  factors: string[]
  weights?: Record<string, number>
  user_priorities?: string[]
}

export interface ComparisonRequest {
  product_ids: string[]
  comparison_criteria?: ComparisonCriteria
}

export interface BackendComparisonRequest {
  product_ids: string[]
  comparison_criteria: ComparisonCriteria
  user_preferences?: Record<string, any>
  include_alternatives?: boolean
}

export interface ProductComparison {
  product: {
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
  scores: Record<string, number>
  strengths: string[]
  weaknesses: string[]
  overall_score: number
}

export interface BackendComparisonResponse {
  products: ProductComparison[]
  comparison_matrix: Record<string, any>
  ai_analysis: string
  recommendation: string
  strengths_weaknesses: Record<string, string[]>
}

export interface ComparisonResponse {
  products: Product[]
  analysis: string
  recommendation?: string
}

export const compareProducts = async (request: ComparisonRequest): Promise<ComparisonResponse> => {
  // Add default comparison criteria if not provided
  const backendRequest: BackendComparisonRequest = {
    product_ids: request.product_ids,
    comparison_criteria: request.comparison_criteria || {
      factors: ['price', 'rating', 'features', 'quality', 'value'],
      user_priorities: ['value', 'quality']
    },
    include_alternatives: false
  }

  const { data } = await advisoryApi.post<BackendComparisonResponse>('/compare', backendRequest)

  // Transform backend response to frontend format
  // Backend returns ProductComparison objects with nested product field
  const products: Product[] = (data.products || []).map((comparison: ProductComparison) => ({
    id: comparison.product.product_id,
    title: comparison.product.title,
    price: comparison.product.price,
    store: comparison.product.store,
    rating: comparison.product.rating,
    image: comparison.product.image_url || '/placeholder-image.jpg',
    url: comparison.product.product_url,
    description: comparison.product.why_recommended,
    inStock: true  // Assume in stock for comparison
  }))

  return {
    products,
    analysis: data.ai_analysis,
    recommendation: data.recommendation
  }
}

export const getProductById = async (id: string): Promise<Product> => {
  const { data } = await discoveryApi.get<any>(`/products/${id}`)

  // Transform backend response to frontend format
  return {
    id: data.id,
    title: data.title,
    description: data.description || '',
    price: data.price,
    originalPrice: data.original_price,
    image: data.image_url || '/placeholder-image.jpg',
    rating: data.rating,
    reviewCount: data.reviews_count,
    store: data.store,
    category: data.category,
    brand: data.brand,
    inStock: data.availability !== 'out_of_stock',
    url: data.product_url
  }
}

import { advisoryApi } from '@/lib/api/axios'
import type { Product } from '@/types/common.types'

export interface BackendTrendingProduct {
  product_id: string
  title: string
  price: number
  store: string
  rating?: number
  image_url?: string
  product_url?: string
  match_score: number
  why_recommended: string
}

export interface BackendTrendingResponse {
  trending_products: BackendTrendingProduct[]
  trending_categories: string[]
  seasonal_recommendations: any[]
  updated_at: string
}

export interface TrendingResponse {
  products: Product[]
  category?: string
}

export const getTrending = async (limit?: number): Promise<TrendingResponse> => {
  const { data } = await advisoryApi.get<BackendTrendingResponse>('/recommendations/trending', {
    params: limit ? { limit } : undefined,
  })

  // Ensure data exists
  if (!data || !data.trending_products) {
    return {
      products: [],
      category: undefined,
    }
  }

  // Convert backend format to frontend Product format
  const trendingProducts = limit ? data.trending_products.slice(0, limit) : data.trending_products
  const products: Product[] = trendingProducts.map((item) => {
    // Debug: Log product IDs to check for duplicates
    if (!item.product_id) {
      console.warn('Product missing product_id:', item)
    }
    return {
      id: item.product_id,
      title: item.title,
      price: item.price,
      store: item.store,
      rating: item.rating,
      image: item.image_url || '/placeholder-image.jpg',
      url: item.product_url,
      description: item.why_recommended,
    }
  })

  return {
    products,
    category: data.trending_categories?.[0],
  }
}

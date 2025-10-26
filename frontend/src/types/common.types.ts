export interface PriceRange {
  min?: number
  max?: number
}

export interface Product {
  id: string
  title: string
  description: string
  price: number
  originalPrice?: number
  image: string
  rating?: number
  reviewCount?: number
  store: string
  category?: string
  brand?: string
  inStock?: boolean
  url?: string
}

export interface SearchFilters {
  priceRange?: PriceRange
  category?: string
  store?: string
  brand?: string
  minRating?: number
  inStockOnly?: boolean
}

export interface UserPreferences {
  budget_range?: PriceRange
  preferred_categories?: string[]
  preferred_stores?: string[]
  tracking_preferences?: {
    allow_activity_tracking?: boolean
  }
}

export type SortOption = 'relevance' | 'price-low' | 'price-high' | 'rating'
export type ViewMode = 'grid' | 'list'

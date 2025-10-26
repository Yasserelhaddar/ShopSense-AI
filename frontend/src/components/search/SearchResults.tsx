import { ProductCard } from '@/components/products/ProductCard'
import { Skeleton } from '@/components/ui/skeleton'
import type { Product } from '@/types/common.types'

interface SearchResultsProps {
  products: Product[]
  isLoading: boolean
  viewMode?: 'grid' | 'list'
}

export function SearchResults({ products, isLoading, viewMode = 'grid' }: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6' : 'space-y-4'}>
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-96" />
        ))}
      </div>
    )
  }

  if (products.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-lg text-muted-foreground">No products found. Try adjusting your search or filters.</p>
      </div>
    )
  }

  return (
    <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6' : 'space-y-4'}>
      {products.map((product) => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  )
}

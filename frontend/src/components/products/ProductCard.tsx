import { Link } from 'react-router-dom'
import { Star, ExternalLink, GitCompare, Check } from 'lucide-react'
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { getProductDetailPath } from '@/lib/constants/routes'
import { useComparisonStore } from '@/features/products/store/comparisonStore'
import type { Product } from '@/types/common.types'

interface ProductCardProps {
  product: Product
}

export function ProductCard({ product }: ProductCardProps) {
  const { addProduct, removeProduct, isSelected, selectedProductIds } = useComparisonStore()
  const selected = isSelected(product.id)
  const maxReached = selectedProductIds.length >= 4 && !selected

  const discount = product.originalPrice
    ? Math.round(((product.originalPrice - product.price) / product.originalPrice) * 100)
    : 0

  const handleCompareToggle = () => {
    if (selected) {
      removeProduct(product.id)
    } else if (!maxReached) {
      addProduct(product.id)
    }
  }

  return (
    <Card className={`w-full max-w-full overflow-hidden hover:shadow-lg transition-shadow h-full flex flex-col ${selected ? 'ring-2 ring-primary' : ''}`}>
      <CardHeader className="p-0">
        <div className="relative w-full h-48 overflow-hidden bg-muted flex items-center justify-center min-w-0">
          <img
            src={product.image || 'https://via.placeholder.com/300'}
            alt={product.title}
            className="w-full h-full object-contain hover:scale-105 transition-transform duration-300"
            loading="lazy"
            onError={(e) => {
              e.currentTarget.src = 'https://via.placeholder.com/300?text=No+Image'
            }}
          />
          {selected && (
            <Badge className="absolute top-2 left-2 bg-primary text-primary-foreground">
              <Check className="w-3 h-3 mr-1" />
              Selected
            </Badge>
          )}
          {discount > 0 && (
            <Badge className="absolute top-2 right-2 bg-destructive text-destructive-foreground">
              -{discount}%
            </Badge>
          )}
          {product.inStock === false && (
            <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
              <Badge variant="secondary">Out of Stock</Badge>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent className="p-4 flex-1 flex flex-col min-w-0">
        <div className="flex items-start justify-between gap-2 mb-2 min-w-0">
          <Badge variant="outline" className="shrink-0">
            {product.store}
          </Badge>
          {product.rating && (
            <div className="flex items-center gap-1 text-sm shrink-0">
              <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
              <span className="font-medium">{product.rating.toFixed(1)}</span>
              {product.reviewCount && (
                <span className="text-muted-foreground">({product.reviewCount})</span>
              )}
            </div>
          )}
        </div>

        <h3 className="font-semibold line-clamp-2 mb-2 flex-1 min-w-0 break-words">{product.title}</h3>

        {product.description && (
          <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
            {product.description}
          </p>
        )}

        <div className="flex items-baseline gap-2 mt-auto">
          <span className="text-2xl font-bold text-primary">
            ${product.price.toFixed(2)}
          </span>
          {product.originalPrice && (
            <span className="text-sm text-muted-foreground line-through">
              ${product.originalPrice.toFixed(2)}
            </span>
          )}
        </div>

        {product.brand && (
          <p className="text-xs text-muted-foreground mt-1">by {product.brand}</p>
        )}
      </CardContent>

      <CardFooter className="p-4 pt-0 flex gap-2 min-w-0">
        <Button asChild className="flex-1 min-w-0">
          <Link to={getProductDetailPath(product.id)}>View Details</Link>
        </Button>
        <Button
          variant={selected ? 'default' : 'outline'}
          size="icon"
          onClick={handleCompareToggle}
          disabled={maxReached}
          title={
            maxReached
              ? 'Maximum 4 products can be compared'
              : selected
              ? 'Remove from comparison'
              : 'Add to comparison'
          }
        >
          {selected ? <Check className="w-4 h-4" /> : <GitCompare className="w-4 h-4" />}
        </Button>
        {product.url && (
          <Button asChild variant="outline" size="icon">
            <a href={product.url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="w-4 h-4" />
            </a>
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}

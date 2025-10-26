import { useParams, Link } from 'react-router-dom'
import { useProduct } from '@/features/products/hooks/useProduct'
import { useComparisonStore } from '@/features/products/store/comparisonStore'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ArrowLeft, ExternalLink, Plus, Check } from 'lucide-react'
import { ROUTES } from '@/lib/constants/routes'

export function ProductDetail() {
  const { id } = useParams<{ id: string }>()
  const { data: product, isLoading, isError } = useProduct(id || '')
  const { selectedProductIds, addProduct, removeProduct, isSelected } =
    useComparisonStore()

  const handleToggleComparison = () => {
    if (!id) return
    if (isSelected(id)) {
      removeProduct(id)
    } else {
      addProduct(id)
    }
  }

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Skeleton className="h-10 w-32 mb-8" />
        <div className="grid md:grid-cols-2 gap-8">
          <Skeleton className="aspect-square w-full" />
          <div className="space-y-4">
            <Skeleton className="h-8 w-3/4" />
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-12 w-full" />
          </div>
        </div>
      </div>
    )
  }

  if (isError || !product) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Link to={ROUTES.SEARCH}>
          <Button variant="ghost" className="mb-8">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Search
          </Button>
        </Link>
        <Card>
          <CardContent className="py-12 text-center">
            <h3 className="text-lg font-semibold mb-2">Product Not Found</h3>
            <p className="text-muted-foreground mb-6">
              The product you're looking for doesn't exist or has been removed.
            </p>
            <Link to={ROUTES.SEARCH}>
              <Button>Go to Search</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  const discount = product.originalPrice
    ? Math.round(
        ((product.originalPrice - product.price) / product.originalPrice) * 100
      )
    : 0

  const inComparison = isSelected(product.id)
  const canAddToComparison = selectedProductIds.length < 4 || inComparison

  return (
    <div className="container mx-auto px-4 py-8">
      <Link to={ROUTES.SEARCH}>
        <Button variant="ghost" className="mb-8">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Search
        </Button>
      </Link>

      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Product Image */}
        <div className="relative">
          <img
            src={product.image}
            alt={product.title}
            className="w-full aspect-square object-cover rounded-lg border"
          />
          {discount > 0 && (
            <Badge className="absolute top-4 right-4 bg-red-500">
              {discount}% OFF
            </Badge>
          )}
          {product.inStock === false && (
            <div className="absolute inset-0 bg-black/60 rounded-lg flex items-center justify-center">
              <Badge variant="secondary" className="text-lg">
                Out of Stock
              </Badge>
            </div>
          )}
        </div>

        {/* Product Info */}
        <div className="space-y-6">
          <div>
            <div className="flex items-start justify-between gap-4 mb-2">
              <h1 className="text-3xl font-bold">{product.title}</h1>
              {product.brand && <Badge variant="outline">{product.brand}</Badge>}
            </div>
            {product.category && (
              <p className="text-muted-foreground">{product.category}</p>
            )}
          </div>

          {/* Price */}
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-bold text-primary">
              ${product.price.toFixed(2)}
            </span>
            {product.originalPrice && (
              <span className="text-xl text-muted-foreground line-through">
                ${product.originalPrice.toFixed(2)}
              </span>
            )}
          </div>

          {/* Rating */}
          {product.rating && (
            <div className="flex items-center gap-2">
              <Badge className="bg-yellow-500">
                {product.rating.toFixed(1)} ⭐
              </Badge>
              {product.reviewCount && (
                <span className="text-sm text-muted-foreground">
                  ({product.reviewCount} reviews)
                </span>
              )}
            </div>
          )}

          {/* Store */}
          <div>
            <p className="text-sm text-muted-foreground mb-1">Sold by</p>
            <Badge variant="outline" className="text-base">
              {product.store}
            </Badge>
          </div>

          {/* Description */}
          {product.description && (
            <div>
              <h2 className="text-lg font-semibold mb-2">Description</h2>
              <p className="text-muted-foreground leading-relaxed">
                {product.description}
              </p>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-3">
            {product.url && (
              <a
                href={product.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block"
              >
                <Button size="lg" className="w-full" disabled={!product.inStock}>
                  {product.inStock === false ? (
                    'Out of Stock'
                  ) : (
                    <>
                      View on {product.store}
                      <ExternalLink className="w-4 h-4 ml-2" />
                    </>
                  )}
                </Button>
              </a>
            )}

            <Button
              variant="outline"
              size="lg"
              className="w-full"
              onClick={handleToggleComparison}
              disabled={!canAddToComparison && !inComparison}
            >
              {inComparison ? (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Added to Comparison
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" />
                  Add to Comparison
                </>
              )}
            </Button>

            {!canAddToComparison && !inComparison && (
              <p className="text-sm text-muted-foreground text-center">
                Maximum 4 products can be compared at once
              </p>
            )}
          </div>

          {selectedProductIds.length >= 2 && (
            <Link to={ROUTES.COMPARISON}>
              <Button variant="secondary" size="lg" className="w-full">
                Go to Comparison ({selectedProductIds.length} products)
              </Button>
            </Link>
          )}
        </div>
      </div>

      {/* Additional Info */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Price History</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Current price: ${product.price.toFixed(2)}
            </p>
            {product.originalPrice && (
              <p className="text-sm text-muted-foreground">
                Original price: ${product.originalPrice.toFixed(2)}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Availability</CardTitle>
          </CardHeader>
          <CardContent>
            {product.inStock !== false ? (
              <Badge className="bg-green-500">In Stock</Badge>
            ) : (
              <Badge variant="secondary">Out of Stock</Badge>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Store Info</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Available at {product.store}
            </p>
            {product.url && (
              <a
                href={product.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline flex items-center gap-1 mt-2"
              >
                Visit store
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

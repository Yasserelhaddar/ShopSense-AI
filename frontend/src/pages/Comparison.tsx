import { useState } from 'react'
import { useComparisonStore } from '@/features/products/store/comparisonStore'
import { useComparison } from '@/features/products/hooks/useComparison'
import { useSearch } from '@/features/search/hooks/useSearch'
import { ProductCard } from '@/components/products/ProductCard'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { X, Loader2, Search } from 'lucide-react'

export function Comparison() {
  const [searchQuery, setSearchQuery] = useState('')
  const { selectedProductIds, removeProduct, clearProducts } = useComparisonStore()
  const { mutate: compare, data: comparisonData, isPending } = useComparison()

  // Search for products to add to comparison
  const { data: searchData, isLoading: isSearching } = useSearch(
    { query: searchQuery, limit: 8 },
    searchQuery.length >= 2
  )

  const handleCompare = () => {
    if (selectedProductIds.length >= 2) {
      compare({ product_ids: selectedProductIds })
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Product Comparison</h1>
        <p className="text-muted-foreground">
          Select products to compare side-by-side with AI-powered analysis
        </p>
      </div>

      {/* Search to add products */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Search Products to Compare</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for products to add to comparison..."
              className="pl-10"
            />
          </div>

          {/* Search Results */}
          {isSearching && searchQuery.length >= 2 && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {searchData && searchData.products.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Found {searchData.total} products - Click the compare icon to add them
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {searchData.products.map((product) => (
                  <ProductCard key={product.id} product={product} />
                ))}
              </div>
            </div>
          )}

          {searchData && searchData.products.length === 0 && searchQuery.length >= 2 && (
            <p className="text-sm text-muted-foreground text-center py-4">
              No products found for "{searchQuery}"
            </p>
          )}

          {searchQuery.length < 2 && (
            <p className="text-sm text-muted-foreground">
              Type at least 2 characters to search for products
            </p>
          )}
        </CardContent>
      </Card>

      {/* Selected products */}
      {selectedProductIds.length > 0 ? (
        <>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-semibold">
                Selected Products ({selectedProductIds.length}/4)
              </h2>
              <Button variant="outline" size="sm" onClick={clearProducts}>
                Clear All
              </Button>
            </div>
            <Button
              onClick={handleCompare}
              disabled={selectedProductIds.length < 2 || isPending}
            >
              {isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Comparing...
                </>
              ) : (
                'Compare Products'
              )}
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {selectedProductIds.map((id) => (
              <Card key={id} className="relative">
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-2 right-2 z-10"
                  onClick={() => removeProduct(id)}
                >
                  <X className="w-4 h-4" />
                </Button>
                <CardContent className="p-4">
                  <div className="aspect-square bg-muted rounded-md mb-3" />
                  <p className="font-medium text-sm">Product ID: {id}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Click to view details
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Comparison Results */}
          {comparisonData && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>AI Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="prose prose-sm max-w-none">
                    <p className="whitespace-pre-wrap">{comparisonData.analysis}</p>
                  </div>
                  {comparisonData.recommendation && (
                    <div className="mt-4 p-4 bg-primary/10 rounded-lg">
                      <p className="font-semibold text-primary mb-2">
                        Recommendation
                      </p>
                      <p>{comparisonData.recommendation}</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Side-by-Side Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-3 font-semibold">Feature</th>
                          {comparisonData.products.map((product) => (
                            <th key={product.id} className="text-left p-3 font-semibold">
                              {product.title}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="p-3 font-medium">Price</td>
                          {comparisonData.products.map((product) => (
                            <td key={product.id} className="p-3">
                              <span className="text-lg font-bold text-primary">
                                ${product.price.toFixed(2)}
                              </span>
                            </td>
                          ))}
                        </tr>
                        <tr className="border-b">
                          <td className="p-3 font-medium">Rating</td>
                          {comparisonData.products.map((product) => (
                            <td key={product.id} className="p-3">
                              {product.rating ? (
                                <Badge>{product.rating.toFixed(1)} ⭐</Badge>
                              ) : (
                                <span className="text-muted-foreground">N/A</span>
                              )}
                            </td>
                          ))}
                        </tr>
                        <tr className="border-b">
                          <td className="p-3 font-medium">Store</td>
                          {comparisonData.products.map((product) => (
                            <td key={product.id} className="p-3">
                              <Badge variant="outline">{product.store}</Badge>
                            </td>
                          ))}
                        </tr>
                        <tr className="border-b">
                          <td className="p-3 font-medium">Brand</td>
                          {comparisonData.products.map((product) => (
                            <td key={product.id} className="p-3">
                              {product.brand || (
                                <span className="text-muted-foreground">N/A</span>
                              )}
                            </td>
                          ))}
                        </tr>
                        <tr>
                          <td className="p-3 font-medium">Availability</td>
                          {comparisonData.products.map((product) => (
                            <td key={product.id} className="p-3">
                              {product.inStock !== false ? (
                                <Badge className="bg-green-500">In Stock</Badge>
                              ) : (
                                <Badge variant="secondary">Out of Stock</Badge>
                              )}
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="py-12 text-center">
            <h3 className="text-lg font-semibold mb-2">No Products Selected</h3>
            <p className="text-muted-foreground mb-6">
              Search for products and add them to your comparison (2-4 products)
            </p>
            <Button asChild>
              <a href="/search">Go to Search</a>
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

import { useState } from 'react'
import { X, Search, Loader2, AlertCircle } from 'lucide-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useComparisonStore } from '@/features/products/store/comparisonStore'
import { useSearch } from '@/features/search/hooks/useSearch'
import { useComparison } from '@/features/products/hooks/useComparison'
import { ProductCard } from '@/components/products/ProductCard'

interface ComparisonModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function ComparisonModal({ open, onOpenChange }: ComparisonModalProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [criteriaInput, setCriteriaInput] = useState('')
  const [userCriteria, setUserCriteria] = useState<string[]>(['price', 'quality', 'value'])
  const { selectedProductIds, removeProduct, clearProducts } = useComparisonStore()

  // Search for products to add
  const { data: searchData, isLoading: isSearching } = useSearch(
    { query: searchQuery, limit: 6 },
    searchQuery.length >= 2
  )

  // Compare products mutation
  const { mutate: compareProducts, data: comparisonData, isPending: isComparing, reset: resetComparison } = useComparison()

  const handleAddCriterion = () => {
    const trimmed = criteriaInput.trim().toLowerCase()
    if (trimmed && !userCriteria.includes(trimmed)) {
      setUserCriteria([...userCriteria, trimmed])
      setCriteriaInput('')
    }
  }

  const handleRemoveCriterion = (criterion: string) => {
    setUserCriteria(userCriteria.filter(c => c !== criterion))
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddCriterion()
    }
  }

  const handleCompare = () => {
    if (selectedProductIds.length >= 2 && userCriteria.length > 0) {
      compareProducts({
        product_ids: selectedProductIds,
        comparison_criteria: {
          factors: userCriteria,
          user_priorities: userCriteria.slice(0, 2)
        }
      })
    }
  }

  const handleClear = () => {
    clearProducts()
    resetComparison()
    setSearchQuery('')
  }

  const handleClose = () => {
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>Product Comparison</span>
            <Button variant="ghost" size="sm" onClick={handleClear}>
              Clear All
            </Button>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Selected Products Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                Selected Products ({selectedProductIds.length}/4)
              </h3>
              {selectedProductIds.length >= 2 && !comparisonData && (
                <Button onClick={handleCompare} disabled={isComparing}>
                  {isComparing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Comparing...
                    </>
                  ) : (
                    'Compare Now'
                  )}
                </Button>
              )}
            </div>

            {selectedProductIds.length === 0 ? (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  No products selected. Search below to add products to compare.
                </AlertDescription>
              </Alert>
            ) : (
              <Card className="p-4">
                <div className="flex flex-wrap gap-2">
                  {selectedProductIds.map((productId) => (
                    <div
                      key={productId}
                      className="flex items-center gap-2 bg-primary/10 px-3 py-1.5 rounded-full"
                    >
                      <span className="text-sm font-medium">{productId.split('_').pop()}</span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-5 w-5 hover:bg-destructive/20"
                        onClick={() => removeProduct(productId)}
                      >
                        <X className="w-3 h-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {selectedProductIds.length === 1 && (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Add at least one more product to compare (2-4 products)
                </AlertDescription>
              </Alert>
            )}
          </div>

          {/* Comparison Criteria Section - Only show before comparison */}
          {!comparisonData && selectedProductIds.length >= 2 && (
            <div className="space-y-3 border-t pt-6">
              <h3 className="text-lg font-semibold">What matters to you?</h3>
              <p className="text-sm text-muted-foreground">
                Add criteria to compare (e.g., "battery life", "durability", "warranty", etc.)
              </p>

              <div className="flex gap-2">
                <Input
                  placeholder="Type a comparison criterion..."
                  value={criteriaInput}
                  onChange={(e) => setCriteriaInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                />
                <Button onClick={handleAddCriterion} variant="outline">
                  Add
                </Button>
              </div>

              {userCriteria.length > 0 && (
                <Card className="p-4">
                  <div className="flex flex-wrap gap-2">
                    {userCriteria.map((criterion) => (
                      <div
                        key={criterion}
                        className="flex items-center gap-2 bg-primary/10 px-3 py-1.5 rounded-full"
                      >
                        <span className="text-sm font-medium capitalize">{criterion}</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-5 w-5 hover:bg-destructive/20"
                          onClick={() => handleRemoveCriterion(criterion)}
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {userCriteria.length === 0 && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Add at least one criterion to compare products
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}

          {/* Comparison Results */}
          {comparisonData && (
            <div className="space-y-4 border-t pt-6">
              <h3 className="text-lg font-semibold">Comparison Results</h3>

              {/* Products with comparison data */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {comparisonData.products.map((product) => (
                  <div key={product.id} className="w-full max-w-full min-w-0 overflow-hidden">
                    <ProductCard product={product} />
                  </div>
                ))}
              </div>

              {/* AI Analysis */}
              {comparisonData.analysis && (
                <Card className="p-6">
                  <h4 className="font-semibold mb-2">AI Analysis</h4>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                    {comparisonData.analysis}
                  </p>
                </Card>
              )}

              {/* Recommendation */}
              {comparisonData.recommendation && (
                <Card className="p-6 bg-primary/5">
                  <h4 className="font-semibold mb-2">Recommendation</h4>
                  <p className="text-sm">{comparisonData.recommendation}</p>
                </Card>
              )}
            </div>
          )}

          {/* Search Section - Only show if not showing results and less than 4 products */}
          {!comparisonData && selectedProductIds.length < 4 && (
            <div className="space-y-3 border-t pt-6">
              <h3 className="text-lg font-semibold">Add More Products</h3>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search for products to compare..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>

              {isSearching && (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                </div>
              )}

              {searchData && searchData.products.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Found {searchData.total} products - Click compare icon to add
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                    {searchData.products.map((product) => (
                      <ProductCard key={product.id} product={product} />
                    ))}
                  </div>
                </div>
              )}

              {searchQuery.length >= 2 && !isSearching && searchData && searchData.products.length === 0 && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    No products found for "{searchQuery}". Try different keywords.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
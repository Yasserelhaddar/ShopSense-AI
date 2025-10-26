import { useState } from 'react'
import { SearchBar } from '@/components/search/SearchBar'
import { SearchResults } from '@/components/search/SearchResults'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useSearch } from '@/features/search/hooks/useSearch'
import { useSearchStore } from '@/features/search/store/searchStore'
import { useDebounce } from '@/hooks/useDebounce'
import { Grid3x3, List, SlidersHorizontal } from 'lucide-react'

export function Search() {
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounce(query, 500)

  const { filters, sortBy, viewMode, setSortBy, setViewMode, clearFilters } = useSearchStore()

  const { data, isLoading, isError, error } = useSearch({
    query: debouncedQuery,
    filters,
  })

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto mb-8">
        <h1 className="text-3xl font-bold mb-6">Product Search</h1>
        <SearchBar value={query} onChange={setQuery} />
      </div>

      {/* Results header */}
      {debouncedQuery && (
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <p className="text-muted-foreground">
              {isLoading ? (
                'Searching...'
              ) : data ? (
                <>
                  Found <span className="font-semibold text-foreground">{data.total}</span> products
                  {data.query && (
                    <>
                      {' '}for <span className="font-semibold text-foreground">"{data.query}"</span>
                    </>
                  )}
                </>
              ) : (
                'No results'
              )}
            </p>

            {/* Active filters badges */}
            {Object.keys(filters).length > 0 && (
              <div className="flex items-center gap-2">
                {Object.entries(filters).map(([key, value]) => {
                  if (!value) return null
                  return (
                    <Badge key={key} variant="secondary">
                      {key}: {JSON.stringify(value)}
                    </Badge>
                  )
                })}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearFilters}
                  className="h-7 text-xs"
                >
                  Clear all
                </Button>
              </div>
            )}
          </div>

          {/* View mode and sort controls */}
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-2 border rounded-md text-sm bg-background"
            >
              <option value="relevance">Most Relevant</option>
              <option value="price-low">Price: Low to High</option>
              <option value="price-high">Price: High to Low</option>
              <option value="rating">Highest Rated</option>
            </select>

            <div className="flex border rounded-md">
              <Button
                variant={viewMode === 'grid' ? 'default' : 'ghost'}
                size="icon"
                onClick={() => setViewMode('grid')}
                className="rounded-r-none"
              >
                <Grid3x3 className="w-4 h-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'ghost'}
                size="icon"
                onClick={() => setViewMode('list')}
                className="rounded-l-none"
              >
                <List className="w-4 h-4" />
              </Button>
            </div>

            <Button variant="outline" size="icon">
              <SlidersHorizontal className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded-md mb-6">
          <p className="font-semibold">Error loading products</p>
          <p className="text-sm">{error instanceof Error ? error.message : 'Something went wrong'}</p>
        </div>
      )}

      {/* Results */}
      {debouncedQuery ? (
        <SearchResults
          products={data?.products || []}
          isLoading={isLoading}
          viewMode={viewMode}
        />
      ) : (
        <div className="text-center py-20">
          <h2 className="text-2xl font-semibold mb-2">Start searching for products</h2>
          <p className="text-muted-foreground">
            Enter a product name or description to find what you're looking for
          </p>
        </div>
      )}
    </div>
  )
}

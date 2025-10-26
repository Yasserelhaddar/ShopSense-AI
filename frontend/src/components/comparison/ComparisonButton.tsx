import { useState } from 'react'
import { GitCompare } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useComparisonStore } from '@/features/products/store/comparisonStore'
import { ComparisonModal } from './ComparisonModal'

export function ComparisonButton() {
  const [open, setOpen] = useState(false)
  const { selectedProductIds } = useComparisonStore()

  if (selectedProductIds.length === 0) {
    return null // Don't show button if no products selected
  }

  return (
    <>
      <Button
        variant="default"
        size="sm"
        onClick={() => setOpen(true)}
        className="relative gap-2"
      >
        <GitCompare className="w-4 h-4" />
        <span className="hidden sm:inline">Compare</span>
        <Badge variant="secondary" className="ml-1 px-1.5 py-0.5 text-xs">
          {selectedProductIds.length}
        </Badge>
      </Button>

      <ComparisonModal open={open} onOpenChange={setOpen} />
    </>
  )
}
import { useState, useEffect } from 'react'
import { useUserPreferences, useUpdatePreferences, useClearHistory } from '@/features/user/hooks/useUserPreferences'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Loader2, Save, Trash2, X } from 'lucide-react'
import type { UserPreferences } from '@/features/user/api/userApi'

const AVAILABLE_CATEGORIES = [
  'Electronics',
  'Clothing',
  'Home & Garden',
  'Sports & Outdoors',
  'Beauty & Personal Care',
  'Books',
  'Toys & Games',
  'Automotive',
  'Food & Grocery',
  'Health & Wellness',
]

const AVAILABLE_STORES = [
  'Amazon',
  'Walmart',
  'Target',
  'Best Buy',
  'eBay',
  'Etsy',
  'Home Depot',
  'Wayfair',
]

export function Settings() {
  const { data: preferences, isLoading } = useUserPreferences()
  const { mutate: updatePrefs, isPending: isUpdating } = useUpdatePreferences()
  const { mutate: clearHist, isPending: isClearing } = useClearHistory()

  const [minBudget, setMinBudget] = useState('')
  const [maxBudget, setMaxBudget] = useState('')
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])
  const [selectedStores, setSelectedStores] = useState<string[]>([])
  const [allowTracking, setAllowTracking] = useState(true)
  const [showClearConfirm, setShowClearConfirm] = useState(false)

  useEffect(() => {
    if (preferences) {
      setMinBudget(preferences.budget_range?.min?.toString() || '')
      setMaxBudget(preferences.budget_range?.max?.toString() || '')
      setSelectedCategories(preferences.preferred_categories || [])
      setSelectedStores(preferences.preferred_stores || [])
      setAllowTracking(preferences.tracking_preferences?.allow_activity_tracking ?? true)
    }
  }, [preferences])

  const handleSave = () => {
    const updatedPreferences: UserPreferences = {
      budget_range: {
        min: minBudget ? parseFloat(minBudget) : undefined,
        max: maxBudget ? parseFloat(maxBudget) : undefined,
      },
      preferred_categories: selectedCategories.length > 0 ? selectedCategories : undefined,
      preferred_stores: selectedStores.length > 0 ? selectedStores : undefined,
      tracking_preferences: {
        allow_activity_tracking: allowTracking,
      },
    }

    updatePrefs(updatedPreferences)
  }

  const handleClearHistory = () => {
    clearHist(undefined, {
      onSuccess: () => {
        setShowClearConfirm(false)
      },
    })
  }

  const toggleCategory = (category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    )
  }

  const toggleStore = (store: string) => {
    setSelectedStores((prev) =>
      prev.includes(store) ? prev.filter((s) => s !== store) : [...prev, store]
    )
  }

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center min-h-[400px]">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Manage your preferences and shopping settings
        </p>
      </div>

      <div className="space-y-6">
        {/* Budget Range */}
        <Card>
          <CardHeader>
            <CardTitle>Budget Range</CardTitle>
            <CardDescription>
              Set your preferred price range for product recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Minimum Price ($)
                </label>
                <Input
                  type="number"
                  value={minBudget}
                  onChange={(e) => setMinBudget(e.target.value)}
                  placeholder="0.00"
                  min="0"
                  step="0.01"
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Maximum Price ($)
                </label>
                <Input
                  type="number"
                  value={maxBudget}
                  onChange={(e) => setMaxBudget(e.target.value)}
                  placeholder="1000.00"
                  min="0"
                  step="0.01"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Preferred Categories */}
        <Card>
          <CardHeader>
            <CardTitle>Preferred Categories</CardTitle>
            <CardDescription>
              Select categories you're most interested in
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_CATEGORIES.map((category) => (
                <Badge
                  key={category}
                  variant={
                    selectedCategories.includes(category) ? 'default' : 'outline'
                  }
                  className="cursor-pointer"
                  onClick={() => toggleCategory(category)}
                >
                  {category}
                  {selectedCategories.includes(category) && (
                    <X className="w-3 h-3 ml-1" />
                  )}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Preferred Stores */}
        <Card>
          <CardHeader>
            <CardTitle>Preferred Stores</CardTitle>
            <CardDescription>
              Select stores you prefer to shop from
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_STORES.map((store) => (
                <Badge
                  key={store}
                  variant={selectedStores.includes(store) ? 'default' : 'outline'}
                  className="cursor-pointer"
                  onClick={() => toggleStore(store)}
                >
                  {store}
                  {selectedStores.includes(store) && (
                    <X className="w-3 h-3 ml-1" />
                  )}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Privacy Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Privacy Settings</CardTitle>
            <CardDescription>
              Control how we use your data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Activity Tracking</p>
                <p className="text-sm text-muted-foreground">
                  Allow us to track your activity to provide better recommendations
                </p>
              </div>
              <Button
                variant={allowTracking ? 'default' : 'outline'}
                onClick={() => setAllowTracking(!allowTracking)}
              >
                {allowTracking ? 'Enabled' : 'Disabled'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Data Management */}
        <Card>
          <CardHeader>
            <CardTitle>Data Management</CardTitle>
            <CardDescription>
              Manage your shopping history and data
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!showClearConfirm ? (
              <Button
                variant="destructive"
                onClick={() => setShowClearConfirm(true)}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear History
              </Button>
            ) : (
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Are you sure you want to clear your entire shopping history? This
                  action cannot be undone.
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="destructive"
                    onClick={handleClearHistory}
                    disabled={isClearing}
                  >
                    {isClearing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        Clearing...
                      </>
                    ) : (
                      'Yes, Clear History'
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setShowClearConfirm(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end">
          <Button onClick={handleSave} disabled={isUpdating} size="lg">
            {isUpdating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Preferences
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}

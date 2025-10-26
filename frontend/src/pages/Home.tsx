import { Link } from 'react-router-dom'
import { ROUTES } from '@/lib/constants/routes'
import { SignedIn, SignedOut, SignInButton } from '@clerk/clerk-react'
import { Search, MessageCircle, BarChart3 } from 'lucide-react'
import { TrendingProducts } from '@/components/recommendations/TrendingProducts'

export function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
            ShopSense AI
          </h1>
          <p className="text-xl text-muted-foreground mb-8">
            Your intelligent shopping assistant powered by AI
          </p>

          <SignedOut>
            <div className="mb-12">
              <SignInButton mode="modal">
                <button className="bg-primary text-primary-foreground px-8 py-3 rounded-lg text-lg font-semibold hover:bg-primary/90 transition-colors">
                  Get Started
                </button>
              </SignInButton>
            </div>
          </SignedOut>

          <SignedIn>
            <div className="flex gap-4 justify-center mb-12">
              <Link
                to={ROUTES.SEARCH}
                className="bg-primary text-primary-foreground px-6 py-3 rounded-lg font-semibold hover:bg-primary/90 transition-colors"
              >
                Start Shopping
              </Link>
              <Link
                to={ROUTES.CONSULTATION}
                className="bg-secondary text-secondary-foreground px-6 py-3 rounded-lg font-semibold hover:bg-secondary/80 transition-colors"
              >
                AI Consultation
              </Link>
            </div>
          </SignedIn>

          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <div className="p-6 bg-card rounded-lg border shadow-sm">
              <Search className="w-12 h-12 text-primary mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">Smart Search</h3>
              <p className="text-muted-foreground">
                Find products with natural language queries and semantic search
              </p>
            </div>

            <div className="p-6 bg-card rounded-lg border shadow-sm">
              <MessageCircle className="w-12 h-12 text-primary mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">AI Consultation</h3>
              <p className="text-muted-foreground">
                Get personalized shopping advice from our AI assistant
              </p>
            </div>

            <div className="p-6 bg-card rounded-lg border shadow-sm">
              <BarChart3 className="w-12 h-12 text-primary mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">Product Comparison</h3>
              <p className="text-muted-foreground">
                Compare products side-by-side with AI-powered analysis
              </p>
            </div>
          </div>

          <SignedIn>
            <div className="mt-16">
              <TrendingProducts limit={8} />
            </div>
          </SignedIn>
        </div>
      </div>
    </div>
  )
}

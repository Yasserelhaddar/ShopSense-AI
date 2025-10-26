import { Link } from 'react-router-dom'
import { SignedIn, SignedOut, SignInButton } from '@clerk/clerk-react'
import { UserButton } from '@/components/auth/UserButton'
import { ROUTES } from '@/lib/constants/routes'
import { Search, MessageCircle, Settings } from 'lucide-react'
import { ComparisonButton } from '@/components/comparison/ComparisonButton'

export function Header() {
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link to={ROUTES.HOME} className="flex items-center space-x-2">
            <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
              ShopSense AI
            </span>
          </Link>

          <SignedIn>
            <nav className="hidden md:flex items-center space-x-6">
              <Link
                to={ROUTES.SEARCH}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
              >
                <Search className="w-4 h-4" />
                Search
              </Link>
              <Link
                to={ROUTES.CONSULTATION}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
              >
                <MessageCircle className="w-4 h-4" />
                Consult
              </Link>
              <Link
                to={ROUTES.SETTINGS}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
              >
                <Settings className="w-4 h-4" />
                Settings
              </Link>
            </nav>
          </SignedIn>

          <div className="flex items-center gap-4">
            <SignedOut>
              <SignInButton mode="modal">
                <button className="bg-primary text-primary-foreground px-4 py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors">
                  Sign In
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <ComparisonButton />
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </div>
    </header>
  )
}

import { useAuth } from '@clerk/clerk-react'
import { Navigate } from 'react-router-dom'
import { ROUTES } from '@/lib/constants/routes'

interface AuthGuardProps {
  children: React.ReactNode
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { isSignedIn, isLoaded } = useAuth()

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (!isSignedIn) {
    return <Navigate to={ROUTES.SIGN_IN} replace />
  }

  return <>{children}</>
}

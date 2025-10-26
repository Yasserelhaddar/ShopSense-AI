import { ClerkProvider, useAuth } from '@clerk/clerk-react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useEffect } from 'react'
import { config } from '@/lib/constants/config'
import { setupAuthInterceptors } from '@/lib/api/axios'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      retry: 3,
      refetchOnWindowFocus: false,
    },
  },
})

function AuthSetup({ children }: { children: React.ReactNode }) {
  const { getToken } = useAuth()

  useEffect(() => {
    setupAuthInterceptors(getToken)
  }, [getToken])

  return <>{children}</>
}

export function Providers({ children }: { children: React.ReactNode }) {
  if (!config.clerk.publishableKey) {
    throw new Error('Missing Clerk Publishable Key')
  }

  return (
    <ClerkProvider publishableKey={config.clerk.publishableKey}>
      <QueryClientProvider client={queryClient}>
        <AuthSetup>
          {children}
          {config.features.enableDevTools && <ReactQueryDevtools />}
        </AuthSetup>
      </QueryClientProvider>
    </ClerkProvider>
  )
}

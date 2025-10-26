import { createBrowserRouter } from 'react-router-dom'
import { MainLayout } from '@/components/layout/MainLayout'
import { AuthGuard } from '@/components/auth/AuthGuard'
import { Home } from '@/pages/Home'
import { Search } from '@/pages/Search'
import { ProductDetail } from '@/pages/ProductDetail'
import { Consultation } from '@/pages/Consultation'
import { Profile } from '@/pages/Profile'
import { Settings } from '@/pages/Settings'
import { Admin } from '@/pages/Admin'
import { NotFound } from '@/pages/NotFound'
import { ROUTES } from '@/lib/constants/routes'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: ROUTES.SEARCH,
        element: (
          <AuthGuard>
            <Search />
          </AuthGuard>
        ),
      },
      {
        path: ROUTES.PRODUCT_DETAIL,
        element: (
          <AuthGuard>
            <ProductDetail />
          </AuthGuard>
        ),
      },
      {
        path: ROUTES.CONSULTATION,
        element: (
          <AuthGuard>
            <Consultation />
          </AuthGuard>
        ),
      },
      {
        path: ROUTES.PROFILE,
        element: (
          <AuthGuard>
            <Profile />
          </AuthGuard>
        ),
      },
      {
        path: ROUTES.SETTINGS,
        element: (
          <AuthGuard>
            <Settings />
          </AuthGuard>
        ),
      },
      {
        path: ROUTES.ADMIN,
        element: (
          <AuthGuard>
            <Admin />
          </AuthGuard>
        ),
      },
    ],
  },
  {
    path: '*',
    element: <NotFound />,
  },
])

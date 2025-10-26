# ShopSense-AI Frontend

Modern React frontend for the ShopSense-AI intelligent shopping assistant platform.

## Overview

The ShopSense-AI frontend provides a user-friendly interface for AI-powered shopping assistance, including product search, personalized consultation, product comparison, and admin system management.

## Tech Stack

- **React 18** + **TypeScript** - Modern React with full type safety
- **Vite** - Fast build tool and dev server
- **React Router v6** - Client-side routing and navigation
- **Tailwind CSS** + **shadcn/ui** - Utility-first styling with accessible components
- **Clerk** - User authentication and management
- **Axios** - HTTP client with interceptors for API communication
- **Lucide React** - Beautiful icon library

## Features

### User Features
✅ **Authentication** - Secure sign-in with Clerk (email/password, OAuth)
✅ **Product Search** - AI-powered semantic product search with filters
✅ **Shopping Consultation** - Conversational AI for personalized product advice
✅ **Product Comparison** - Detailed multi-criteria comparison with AI analysis
✅ **Product Details** - Comprehensive product information with images and ratings
✅ **User Profile** - Manage shopping preferences and account settings
✅ **Activity Tracking** - Optional activity history with privacy controls

### Admin Features (Role-Based Access)
✅ **Data Collection Management** - Trigger and monitor product collection jobs
✅ **Training Data Generation** - Generate synthetic training data for AI models
✅ **Model Training** - Start and monitor model fine-tuning jobs
✅ **System Monitoring** - View health status and statistics across all services

## Quick Start

### Prerequisites

- **Node.js 18+** - JavaScript runtime
- **npm** or **yarn** - Package manager
- **Clerk Account** - For authentication (sign up at [clerk.com](https://clerk.com))
- **Running Backend Services** - Advisory, Discovery, and Knowledge engines

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create `.env.local` file with your Clerk keys:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```bash
# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your-clerk-publishable-key

# Backend API URLs
VITE_ADVISORY_API_URL=http://localhost:8003/api/v1
VITE_DISCOVERY_API_URL=http://localhost:8002/api/v1
VITE_KNOWLEDGE_API_URL=http://localhost:8001/api/v1
```

### Run Development Server

```bash
npm run dev
```

Visit http://localhost:5173

### Build for Production

```bash
npm run build
npm run preview  # Preview production build
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # App configuration
│   │   ├── App.tsx            # Root component
│   │   ├── providers.tsx      # Context providers
│   │   └── router.tsx         # Route definitions
│   ├── pages/                 # Page components
│   │   ├── Home.tsx           # Landing page
│   │   ├── Search.tsx         # Product search
│   │   ├── Consultation.tsx   # AI consultation
│   │   ├── Comparison.tsx     # Product comparison
│   │   ├── ProductDetail.tsx  # Product details
│   │   ├── Profile.tsx        # User profile
│   │   ├── Settings.tsx       # User settings
│   │   └── Admin.tsx          # Admin dashboard
│   ├── components/            # Reusable components
│   │   ├── layout/           # Layout components
│   │   ├── auth/             # Authentication components
│   │   ├── products/         # Product components
│   │   ├── consultation/     # Consultation components
│   │   ├── comparison/       # Comparison components
│   │   └── ui/               # shadcn/ui components
│   ├── features/             # Feature modules
│   │   ├── products/         # Product API & hooks
│   │   ├── consultation/     # Consultation API & hooks
│   │   └── admin/            # Admin API & hooks
│   ├── lib/                  # Utilities
│   │   ├── api/             # API clients
│   │   ├── constants/       # App constants
│   │   └── utils.ts         # Helper functions
│   ├── types/               # TypeScript types
│   └── styles/              # Global styles
├── public/                  # Static assets
└── index.html              # HTML entry point
```

## Authentication Setup

### Clerk Configuration

1. **Create Clerk Application**: Sign up and create app at [clerk.com](https://clerk.com)
2. **Get API Keys**: Copy publishable key from Clerk dashboard
3. **Configure Session Token**:
   - Go to Configure → Sessions → Customize session token
   - Add claims:
   ```json
   {
     "email": "{{user.primary_email_address}}",
     "metadata": "{{user.public_metadata}}"
   }
   ```
4. **Enable OAuth Providers** (optional): Google, GitHub, etc.

### Admin Role Setup

To grant admin access to a user:

1. Go to Clerk Dashboard → Users
2. Select user
3. Click "Metadata" tab
4. Add to "Public metadata":
   ```json
   {
     "role": "admin"
   }
   ```
5. Save changes

Admin users will see an "Admin" link in the header and can access the admin dashboard at `/admin`.

## API Integration

### API Clients

The app uses Axios clients with authentication interceptors:

```typescript
// Advisory Engine API
import { advisoryApi } from '@/lib/api/axios'
const response = await advisoryApi.post('/search', { query: '...' })

// Discovery Engine API
import { discoveryApi } from '@/lib/api/axios'
const response = await discoveryApi.get('/products/search?query=...')
```

### Authentication

All API requests automatically include the Clerk JWT token via Axios interceptors configured in `src/lib/api/axios.ts`.

## Development

### Code Style

- **TypeScript**: Strict mode enabled with comprehensive type checking
- **ESLint**: Configured for React and TypeScript best practices
- **Prettier**: Automatic code formatting (run with `npm run format`)

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run format       # Format code with Prettier
npm run type-check   # Run TypeScript compiler check
```

### Testing

```bash
npm run test         # Run tests (when configured)
npm run test:watch   # Run tests in watch mode
```

## Deployment

### Build Configuration

The production build is optimized with:
- Code splitting and lazy loading
- Tree shaking for minimal bundle size
- Asset optimization and compression
- Source maps for debugging

### Environment Variables for Production

```bash
# Production .env
VITE_CLERK_PUBLISHABLE_KEY=pk_live_your-production-key
VITE_ADVISORY_API_URL=https://api.yourdomain.com/advisory/api/v1
VITE_DISCOVERY_API_URL=https://api.yourdomain.com/discovery/api/v1
VITE_KNOWLEDGE_API_URL=https://api.yourdomain.com/knowledge/api/v1
```

### Deploy to Vercel (Recommended)

```bash
npm install -g vercel
vercel --prod
```

Configure environment variables in Vercel dashboard.

### Deploy to Netlify

```bash
npm run build
netlify deploy --prod --dir=dist
```

### Deploy with Docker

```bash
# Build production image
docker build -t shopsense-frontend:latest .

# Run container
docker run -d \
  -p 80:80 \
  -e VITE_CLERK_PUBLISHABLE_KEY=pk_live_... \
  -e VITE_ADVISORY_API_URL=https://api.yourdomain.com/advisory/api/v1 \
  --name frontend \
  shopsense-frontend:latest
```

## Features Deep Dive

### Product Search
- Semantic search powered by vector embeddings
- Filter by price, category, brand, store
- Real-time results with AI-enhanced relevance

### Shopping Consultation
- Conversational AI interface
- Product-aware recommendations
- Context retention across conversation
- Suggested questions for guidance

### Product Comparison
- Side-by-side comparison of multiple products
- AI-powered analysis and recommendations
- Detailed strengths/weaknesses breakdown
- Overall scores and rankings

### Admin Dashboard
**Access**: Requires admin role in Clerk public metadata

**Capabilities**:
- **Data Collection**: Trigger product scraping from multiple stores
- **Training Data**: Generate synthetic training datasets
- **Model Training**: Start fine-tuning jobs with custom parameters
- **System Monitoring**: View service health and statistics

## Troubleshooting

### Common Issues

**Authentication not working**:
- Verify Clerk publishable key in `.env.local`
- Check session token claims are configured in Clerk
- Ensure backend has correct Clerk issuer URL

**API calls failing**:
- Verify backend services are running
- Check API URLs in `.env.local`
- Inspect browser console for CORS errors

**Admin features not visible**:
- Verify user has `role: "admin"` in Clerk public metadata
- Check session token includes metadata claim
- Clear browser cache and re-login

**Build errors**:
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Clear Vite cache: `rm -rf node_modules/.vite`
- Verify Node.js version is 18+

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "feat: add amazing feature"`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open pull request

## License

MIT License - see main project LICENSE file for details.

---

**Part of the ShopSense-AI Platform** | [Main Documentation](../README.md)
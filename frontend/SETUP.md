# ShopSense-AI Frontend - Quick Setup Guide

## Prerequisites

- Node.js 18+ installed
- Backend services running (see below)
- Clerk account for authentication

## Setup Steps

### 1. Backend Services (Run First!)

Navigate to each service and start them:

```bash
# Terminal 1 - Knowledge Engine (Port 8001)
cd services/knowledge_engine
uv run main

# Terminal 2 - Discovery Engine (Port 8002)
cd services/discovery_engine
uv run main

# Terminal 3 - Advisory Engine (Port 8003)
cd services/advisory_engine
uv run main
```

**Expected Output:**
- Knowledge Engine: http://localhost:8001
- Discovery Engine: http://localhost:8002
- Advisory Engine: http://localhost:8003

### 2. Clerk Authentication Setup

1. Go to [Clerk Dashboard](https://dashboard.clerk.com/)
2. Create a new application (or use existing)
3. Get your **Publishable Key** from the API Keys section
4. Copy the key - it should look like: `pk_test_...` or `pk_live_...`

### 3. Frontend Environment Variables

```bash
cd frontend

# Copy the example env file
cp .env.example .env

# Edit .env and add your Clerk key
# VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_actual_key_here
```

Your `.env` file should look like this:

```env
# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_actual_key_here

# Backend API URLs (default - no need to change if using default ports)
VITE_ADVISORY_API_URL=http://localhost:8003/api/v1
VITE_DISCOVERY_API_URL=http://localhost:8002/api/v1
VITE_KNOWLEDGE_API_URL=http://localhost:8001/api/v1
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 5. Start Frontend Development Server

```bash
npm run dev
```

The frontend will be available at: **http://localhost:5173**

## Verification Checklist

Before using the app, verify:

- [ ] All 3 backend services are running (ports 8001, 8002, 8003)
- [ ] Frontend `.env` file exists with valid Clerk key
- [ ] Frontend dev server is running on port 5173
- [ ] You can access http://localhost:5173 in your browser

## Quick Test

1. Open http://localhost:5173
2. Click "Get Started" or "Sign In"
3. Sign in with Clerk (email or social)
4. Try the Search page - type a product query
5. Try the Consultation page - ask the AI assistant

## Troubleshooting

### "Authentication Required" errors
- Check your Clerk publishable key is correct in `.env`
- Restart the frontend dev server after changing `.env`

### "Network Error" or API failures
- Verify all 3 backend services are running
- Check the terminal logs for each service
- Ensure ports 8001, 8002, 8003 are not blocked

### Build errors
- Delete `node_modules` and run `npm install` again
- Clear npm cache: `npm cache clean --force`

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_CLERK_PUBLISHABLE_KEY` | ✅ Yes | - | Clerk authentication publishable key |
| `VITE_ADVISORY_API_URL` | No | `http://localhost:8003/api/v1` | Advisory Engine API URL |
| `VITE_DISCOVERY_API_URL` | No | `http://localhost:8002/api/v1` | Discovery Engine API URL |
| `VITE_KNOWLEDGE_API_URL` | No | `http://localhost:8001/api/v1` | Knowledge Engine API URL |

## Production Build

To create a production build:

```bash
npm run build
```

Built files will be in the `dist/` directory.

To preview the production build:

```bash
npm run preview
```

## Common Commands

```bash
# Development
npm run dev              # Start dev server

# Build
npm run build           # Production build
npm run preview         # Preview production build

# Linting
npm run lint            # Check code quality
```

## Need Help?

- **Backend Issues**: Check the README in each service directory
- **Frontend Issues**: Check browser console and terminal logs
- **Authentication Issues**: Verify Clerk dashboard settings

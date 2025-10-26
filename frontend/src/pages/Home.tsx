import { Link } from 'react-router-dom'
import { ROUTES } from '@/lib/constants/routes'
import { SignedIn, SignedOut, SignInButton } from '@clerk/clerk-react'
import { Search, MessageCircle, BarChart3, Sparkles, Zap, TrendingUp } from 'lucide-react'
import { TrendingProducts } from '@/components/recommendations/TrendingProducts'
import { Canvas } from '@react-three/fiber'
import { FloatingShoppingBag } from '@/components/home/FloatingShoppingBag'
import { AnimatedBackground } from '@/components/home/AnimatedBackground'
import { motion } from 'framer-motion'
import { OrbitControls } from '@react-three/drei'

const fadeInUp = {
  hidden: { opacity: 0, y: 60 },
  visible: { opacity: 1, y: 0 },
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
}

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: {
      type: 'spring',
      damping: 15,
      stiffness: 100,
    }
  },
}

export function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />

      <div className="container mx-auto px-4 py-16 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
          {/* Left side - Content */}
          <motion.div
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="text-left"
          >
            <motion.div
              variants={fadeInUp}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 backdrop-blur-sm mb-6"
            >
              <Sparkles className="w-4 h-4 text-blue-500" />
              <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                Powered by AI
              </span>
            </motion.div>

            <motion.h1
              variants={fadeInUp}
              className="text-6xl md:text-7xl font-bold mb-6 leading-tight"
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 bg-[length:200%_auto] animate-gradient">
                ShopSense AI
              </span>
            </motion.h1>

            <motion.p
              variants={fadeInUp}
              className="text-xl md:text-2xl text-muted-foreground mb-4"
            >
              Your intelligent shopping assistant
            </motion.p>

            <motion.p
              variants={fadeInUp}
              className="text-lg text-muted-foreground/80 mb-8 max-w-lg"
            >
              Discover products with AI-powered semantic search, get personalized recommendations, and make smarter shopping decisions.
            </motion.p>

            <SignedOut>
              <motion.div variants={fadeInUp} className="mb-12">
                <SignInButton mode="modal">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="group relative px-8 py-4 rounded-xl text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40 transition-all duration-300"
                  >
                    <span className="relative z-10 flex items-center gap-2">
                      Get Started Free
                      <Zap className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </span>
                    <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 opacity-0 group-hover:opacity-100 blur transition-opacity" />
                  </motion.button>
                </SignInButton>
              </motion.div>
            </SignedOut>

            <SignedIn>
              <motion.div
                variants={fadeInUp}
                className="flex flex-wrap gap-4 mb-12"
              >
                <Link to={ROUTES.SEARCH}>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-6 py-3 rounded-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg hover:shadow-xl transition-all"
                  >
                    Start Shopping
                  </motion.button>
                </Link>
                <Link to={ROUTES.CONSULTATION}>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-6 py-3 rounded-xl font-semibold bg-white/10 backdrop-blur-md border border-white/20 hover:bg-white/20 transition-all"
                  >
                    AI Consultation
                  </motion.button>
                </Link>
              </motion.div>
            </SignedIn>

            {/* Stats */}
            <motion.div
              variants={fadeInUp}
              className="flex flex-wrap gap-8"
            >
              <div>
                <div className="text-3xl font-bold text-blue-600">10K+</div>
                <div className="text-sm text-muted-foreground">Products</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-600">AI-Powered</div>
                <div className="text-sm text-muted-foreground">Search</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-blue-600">24/7</div>
                <div className="text-sm text-muted-foreground">Assistant</div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right side - 3D Shopping Bag */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.3 }}
            className="relative h-[500px] lg:h-[600px]"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-3xl backdrop-blur-3xl" />
            <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
              <FloatingShoppingBag />
              <OrbitControls
                enableZoom={false}
                enablePan={false}
                autoRotate
                autoRotateSpeed={0.5}
              />
            </Canvas>
          </motion.div>
        </div>

        {/* Features Section with Glassmorphism Cards */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={staggerContainer}
          className="grid md:grid-cols-3 gap-6 mb-16"
        >
          {[
            {
              icon: Search,
              title: 'Smart Search',
              description: 'Find products with natural language queries and semantic search powered by AI',
              gradient: 'from-blue-500 to-cyan-500',
            },
            {
              icon: MessageCircle,
              title: 'AI Consultation',
              description: 'Get personalized shopping advice from our intelligent AI assistant',
              gradient: 'from-purple-500 to-pink-500',
            },
            {
              icon: BarChart3,
              title: 'Product Comparison',
              description: 'Compare products side-by-side with AI-powered analysis and recommendations',
              gradient: 'from-orange-500 to-red-500',
            },
          ].map((feature, index) => (
            <motion.div
              key={index}
              variants={scaleIn}
              whileHover={{ y: -10, scale: 1.02 }}
              className="group relative p-8 rounded-2xl bg-white/5 backdrop-blur-lg border border-white/10 hover:border-white/20 shadow-xl hover:shadow-2xl transition-all duration-300"
            >
              {/* Gradient border effect on hover */}
              <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity`} />

              {/* Icon with gradient background */}
              <div className={`relative w-14 h-14 mb-6 rounded-xl bg-gradient-to-br ${feature.gradient} p-3 shadow-lg`}>
                <feature.icon className="w-full h-full text-white" />
              </div>

              <h3 className="text-2xl font-bold mb-3 relative z-10">
                {feature.title}
              </h3>
              <p className="text-muted-foreground relative z-10">
                {feature.description}
              </p>

              {/* Hover effect decoration */}
              <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <TrendingUp className="w-5 h-5 text-blue-500" />
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Trending Products */}
        <SignedIn>
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <TrendingProducts limit={8} />
          </motion.div>
        </SignedIn>
      </div>

      <style>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  )
}
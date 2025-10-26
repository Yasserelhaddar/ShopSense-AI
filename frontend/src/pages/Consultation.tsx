import { useState, useRef, useEffect } from 'react'
import { MessageBubble } from '@/components/consultation/MessageBubble'
import { MessageInput } from '@/components/consultation/MessageInput'
import { SuggestedQuestions } from '@/components/consultation/SuggestedQuestions'
import { ProductCard } from '@/components/products/ProductCard'
import { useConsultation } from '@/features/consultation/hooks/useConsultation'
import { Card } from '@/components/ui/card'
import { Loader2 } from 'lucide-react'
import type { Product } from '@/types/common.types'
import type { ProductRecommendation } from '@/features/consultation/api/consultationApi'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function Consultation() {
  const [messages, setMessages] = useState<Message[]>([])
  const [recommendations, setRecommendations] = useState<Product[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { mutate: getAdvice, isPending } = useConsultation()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = (content: string) => {
    // Add user message
    const userMessage: Message = { role: 'user', content }
    setMessages((prev) => [...prev, userMessage])

    // Get AI advice
    getAdvice(
      { query: content },
      {
        onSuccess: (data) => {
          const assistantMessage: Message = {
            role: 'assistant',
            content: data.advice,
          }
          setMessages((prev) => [...prev, assistantMessage])

          // Convert backend ProductRecommendation to frontend Product
          if (data.product_recommendations && data.product_recommendations.length > 0) {
            const products: Product[] = data.product_recommendations.map(
              (rec: ProductRecommendation) => ({
                id: rec.product_id,
                title: rec.title,
                price: rec.price,
                store: rec.store,
                rating: rec.rating,
                image: rec.image_url || '/placeholder-image.jpg',
                url: rec.product_url,
                description: rec.why_recommended,
              })
            )
            setRecommendations(products)
          }
        },
        onError: (error) => {
          const errorMessage: Message = {
            role: 'assistant',
            content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Something went wrong'}. Please try again.`,
          }
          setMessages((prev) => [...prev, errorMessage])
        },
      }
    )
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6">AI Shopping Consultation</h1>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Chat area */}
        <div className="lg:col-span-2">
          <Card className="flex flex-col h-[600px]">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 ? (
                <div className="text-center py-12">
                  <h2 className="text-2xl font-semibold mb-4">
                    How can I help you today?
                  </h2>
                  <p className="text-muted-foreground mb-8">
                    I'm your AI shopping assistant. Ask me anything about products!
                  </p>
                  <SuggestedQuestions onSelect={handleSendMessage} />
                </div>
              ) : (
                <>
                  {messages.map((message, index) => (
                    <MessageBubble key={index} message={message} />
                  ))}
                  {isPending && (
                    <div className="flex justify-start">
                      <div className="bg-muted rounded-lg px-4 py-3 flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm">Thinking...</span>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Input */}
            <div className="border-t p-4">
              <MessageInput onSend={handleSendMessage} isLoading={isPending} />
            </div>
          </Card>
        </div>

        {/* Recommendations sidebar */}
        <div className="lg:col-span-1">
          {recommendations.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Recommended Products</h3>
              <div className="space-y-4">
                {recommendations.slice(0, 3).map((product) => (
                  <ProductCard key={product.id} product={product} />
                ))}
              </div>
            </div>
          ) : (
            <Card className="p-6 text-center text-muted-foreground">
              <p>Product recommendations will appear here based on your conversation.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

import { Button } from '@/components/ui/button'

const SUGGESTED_QUESTIONS = [
  'What laptop should I buy for programming?',
  'Best noise-canceling headphones under $200?',
  'Help me choose a gaming monitor',
  'Which smartphone has the best camera?',
]

interface SuggestedQuestionsProps {
  onSelect: (question: string) => void
}

export function SuggestedQuestions({ onSelect }: SuggestedQuestionsProps) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground font-medium">Try asking:</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {SUGGESTED_QUESTIONS.map((question) => (
          <Button
            key={question}
            variant="outline"
            onClick={() => onSelect(question)}
            className="justify-start text-left h-auto py-3 px-4"
          >
            {question}
          </Button>
        ))}
      </div>
    </div>
  )
}

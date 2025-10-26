import { useMutation } from '@tanstack/react-query'
import { getAdvice, type AdviceRequest } from '../api/consultationApi'

export function useConsultation() {
  return useMutation({
    mutationFn: (request: AdviceRequest) => getAdvice(request),
  })
}

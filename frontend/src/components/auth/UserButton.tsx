import { UserButton as ClerkUserButton } from '@clerk/clerk-react'

export function UserButton() {
  return (
    <ClerkUserButton
      appearance={{
        elements: {
          avatarBox: 'w-10 h-10',
        },
      }}
    />
  )
}

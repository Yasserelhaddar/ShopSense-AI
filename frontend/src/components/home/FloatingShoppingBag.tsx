import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Float, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

function ShoppingBagMesh() {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.3
    }
  })

  return (
    <Float
      speed={2}
      rotationIntensity={0.5}
      floatIntensity={1}
      floatingRange={[-0.1, 0.1]}
    >
      <mesh ref={meshRef}>
        {/* Shopping bag body */}
        <group>
          {/* Main bag */}
          <mesh position={[0, 0, 0]}>
            <boxGeometry args={[1.5, 2, 1]} />
            <meshStandardMaterial
              color="#3b82f6"
              metalness={0.3}
              roughness={0.4}
            />
          </mesh>

          {/* Handles */}
          <mesh position={[-0.5, 1.2, 0]}>
            <torusGeometry args={[0.3, 0.08, 16, 32, Math.PI]} />
            <meshStandardMaterial
              color="#2563eb"
              metalness={0.5}
              roughness={0.3}
            />
          </mesh>
          <mesh position={[0.5, 1.2, 0]}>
            <torusGeometry args={[0.3, 0.08, 16, 32, Math.PI]} />
            <meshStandardMaterial
              color="#2563eb"
              metalness={0.5}
              roughness={0.3}
            />
          </mesh>

          {/* Shopping cart icon on front */}
          <mesh position={[0, 0, 0.51]}>
            <boxGeometry args={[0.6, 0.6, 0.1]} />
            <meshStandardMaterial
              color="#60a5fa"
              emissive="#3b82f6"
              emissiveIntensity={0.2}
            />
          </mesh>

          {/* Sparkle effects */}
          {[...Array(8)].map((_, i) => (
            <mesh
              key={i}
              position={[
                Math.cos((i / 8) * Math.PI * 2) * 1.5,
                Math.sin((i / 8) * Math.PI * 2) * 1.5,
                0
              ]}
            >
              <sphereGeometry args={[0.05, 8, 8]} />
              <meshStandardMaterial
                color="#60a5fa"
                emissive="#3b82f6"
                emissiveIntensity={0.5}
              />
            </mesh>
          ))}
        </group>
      </mesh>
    </Float>
  )
}

export function FloatingShoppingBag() {
  return (
    <>
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <ShoppingBagMesh />
    </>
  )
}
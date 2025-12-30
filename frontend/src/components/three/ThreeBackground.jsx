import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

function FloatingParticles({ count = 100 }) {
    const mesh = useRef()
    const light = useRef()

    const particles = useMemo(() => {
        const temp = []
        for (let i = 0; i < count; i++) {
            const time = Math.random() * 100
            const factor = 20 + Math.random() * 100
            const speed = 0.005 + Math.random() / 200
            const x = Math.random() * 100 - 50
            const y = Math.random() * 100 - 50
            const z = Math.random() * 100 - 50

            temp.push({ time, factor, speed, x, y, z })
        }
        return temp
    }, [count])

    const dummy = useMemo(() => new THREE.Object3D(), [])

    useFrame(() => {
        particles.forEach((particle, i) => {
            let { time, factor, speed, x, y, z } = particle
            time = particle.time += speed

            dummy.position.set(
                x + Math.cos((time / 10) * factor) * 2,
                y + Math.sin((time / 10) * factor) * 2,
                z + Math.sin((time / 10) * factor) * 2
            )

            const scale = Math.cos(time) * 0.3 + 0.7
            dummy.scale.set(scale, scale, scale)
            dummy.updateMatrix()
            mesh.current.setMatrixAt(i, dummy.matrix)
        })
        mesh.current.instanceMatrix.needsUpdate = true
    })

    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight ref={light} position={[0, 0, 0]} intensity={0.5} color="#0ea5e9" />
            <instancedMesh ref={mesh} args={[null, null, count]}>
                <sphereGeometry args={[0.15, 16, 16]} />
                <meshStandardMaterial
                    color="#0ea5e9"
                    transparent
                    opacity={0.6}
                    emissive="#06b6d4"
                    emissiveIntensity={0.3}
                />
            </instancedMesh>
        </>
    )
}

function FloatingRings() {
    const group = useRef()

    useFrame((state) => {
        group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.1) * 0.1
        group.current.rotation.y += 0.002
    })

    return (
        <group ref={group} position={[15, 0, -20]}>
            <mesh>
                <torusGeometry args={[8, 0.3, 16, 100]} />
                <meshStandardMaterial
                    color="#14b8a6"
                    transparent
                    opacity={0.3}
                    wireframe
                />
            </mesh>
            <mesh rotation={[Math.PI / 3, 0, 0]}>
                <torusGeometry args={[6, 0.2, 16, 100]} />
                <meshStandardMaterial
                    color="#0ea5e9"
                    transparent
                    opacity={0.3}
                    wireframe
                />
            </mesh>
            <mesh rotation={[Math.PI / 6, Math.PI / 4, 0]}>
                <torusGeometry args={[10, 0.15, 16, 100]} />
                <meshStandardMaterial
                    color="#38bdf8"
                    transparent
                    opacity={0.2}
                    wireframe
                />
            </mesh>
        </group>
    )
}

export default function ThreeBackground() {
    return (
        <div className="three-background">
            <Canvas
                camera={{ position: [0, 0, 30], fov: 75 }}
                style={{ background: 'transparent' }}
            >
                <fog attach="fog" args={['#f8fafc', 30, 100]} />
                <FloatingParticles count={80} />
                <FloatingRings />
            </Canvas>
        </div>
    )
}

"use client";

import { Float, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { Dashboard, Stage } from "../lib/api";

const positions: Record<Stage, [number, number, number]> = {
  sediment: [0, 1.15, 0],
  carbon: [0, 0, 0],
  ro: [0, -1.15, 0],
};

function FlowParticles() {
  const ref = useRef<THREE.Points>(null);
  const points = useMemo(() => {
    const values = new Float32Array(180 * 3);
    for (let i = 0; i < 180; i += 1) {
      values[i * 3] = (Math.random() - 0.5) * 0.65;
      values[i * 3 + 1] = Math.random() * 4 - 2;
      values[i * 3 + 2] = (Math.random() - 0.5) * 0.65;
    }
    return values;
  }, []);
  useFrame((_, delta) => {
    if (!ref.current) return;
    ref.current.rotation.y += delta * 0.08;
    const position = ref.current.geometry.attributes.position;
    for (let i = 1; i < position.count * 3; i += 3) {
      position.array[i] = (position.array[i] as number) - delta * 0.35;
      if ((position.array[i] as number) < -2) position.array[i] = 2;
    }
    position.needsUpdate = true;
  });
  return (
    <points ref={ref}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[points, 3]} /></bufferGeometry>
      <pointsMaterial color="#6ee7e0" size={0.035} transparent opacity={0.75} />
    </points>
  );
}

function Cartridge({ stage, progress, selected, onSelect, health }: {
  stage: Stage; progress: number; selected: boolean; onSelect: () => void; health: string;
}) {
  const color = health !== "ok" ? "#ff7777" : progress > 0.75 ? "#f5be62" : "#54e4d7";
  return (
    <Float speed={1.2} rotationIntensity={selected ? 0.14 : 0.04} floatIntensity={selected ? 0.18 : 0.05}>
      <group position={positions[stage]} onClick={onSelect}>
        <mesh>
          <capsuleGeometry args={[0.48, 0.7, 8, 24]} />
          <meshPhysicalMaterial color="#10253a" metalness={0.45} roughness={0.25} transmission={0.18} thickness={0.4} />
        </mesh>
        <mesh position={[0, 0, 0.49]}>
          <planeGeometry args={[0.54, 0.58]} />
          <meshBasicMaterial color={color} transparent opacity={selected ? 0.9 : 0.65} />
        </mesh>
        <mesh position={[0, 0.36, 0.54]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>
      </group>
    </Float>
  );
}

function Scene({ dashboard, selected, onSelect }: { dashboard: Dashboard | null; selected: Stage; onSelect: (stage: Stage) => void }) {
  return (
    <>
      <PerspectiveCamera makeDefault position={[4.2, 1.6, 6]} fov={38} />
      <ambientLight intensity={0.7} color="#b9efff" />
      <pointLight position={[3, 4, 4]} intensity={3} color="#70dfe0" />
      <pointLight position={[-3, -2, 2]} intensity={2} color="#617dff" />
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[1.05, 1.05, 3.9, 48, 1, true]} />
        <meshPhysicalMaterial color="#102d43" transparent opacity={0.24} transmission={0.75} roughness={0.12} side={THREE.DoubleSide} />
      </mesh>
      {(["sediment", "carbon", "ro"] as Stage[]).map((stage) => (
        <Cartridge key={stage} stage={stage} selected={selected === stage} onSelect={() => onSelect(stage)}
          progress={dashboard?.stages[stage]?.condition.progress || 0}
          health={dashboard?.stages[stage]?.sensor_health || "ok"} />
      ))}
      <FlowParticles />
      <OrbitControls enablePan={false} minDistance={4.5} maxDistance={8} enableDamping />
    </>
  );
}

export default function PurifierScene({ dashboard, selected, onSelect }: { dashboard: Dashboard | null; selected: Stage; onSelect: (stage: Stage) => void }) {
  return (
    <Canvas dpr={[1, 1.5]} gl={{ antialias: true }} fallback={<div className="scene-fallback">3D view unavailable. Text forecast remains available.</div>}>
      <Scene dashboard={dashboard} selected={selected} onSelect={onSelect} />
    </Canvas>
  );
}

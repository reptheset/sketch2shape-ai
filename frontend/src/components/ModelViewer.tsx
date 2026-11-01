"use client";

// Interactive 3D preview of the generated .glb, built with React Three Fiber + drei.
// Loaded with next/dynamic (ssr: false) because WebGL only exists in the browser.
//
// - Lighting is a small studio made of drei <Lightformer>s, so no HDR file is downloaded
//   from the internet (works offline at the venue).
// - Materials are rendered double-sided: single-view models are often open at the back or
//   bottom, and back-faces would otherwise show as holes.
// - <Bounds> frames the model automatically, whatever its size.

import { Canvas } from "@react-three/fiber";
import {
  Bounds,
  ContactShadows,
  Environment,
  Html,
  Lightformer,
  OrbitControls,
  useBounds,
  useGLTF,
  useProgress,
} from "@react-three/drei";
import { Component, Suspense, useEffect, type ReactNode } from "react";
import * as THREE from "three";

export type ModelViewerProps = {
  url: string;
  autoRotate: boolean;
  wireframe: boolean;
  /** Increment to animate the camera back to the default framing. */
  resetSignal: number;
};

function Model({ url, wireframe }: { url: string; wireframe: boolean }) {
  // Second argument false: our GLBs are not Draco-compressed, so never fetch the Draco decoder.
  const { scene } = useGLTF(url, false);

  useEffect(() => {
    scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        const m = material as THREE.MeshStandardMaterial;
        m.side = THREE.DoubleSide;
        m.wireframe = wireframe;
        m.needsUpdate = true;
      }
    });
  }, [scene, wireframe]);

  return <primitive object={scene} />;
}

function Refit({ signal }: { signal: number }) {
  const bounds = useBounds();
  useEffect(() => {
    if (signal > 0) bounds.refresh().clip().fit();
  }, [signal, bounds]);
  return null;
}

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="flex flex-col items-center gap-2 whitespace-nowrap text-sm font-medium text-slate-500">
        <div className="h-1.5 w-40 overflow-hidden rounded-full bg-slate-200">
          <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} />
        </div>
        Loading model… {Math.round(progress)}%
      </div>
    </Html>
  );
}

function StudioLights() {
  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[3, 5, 4]} intensity={1.1} castShadow />
      <Environment resolution={256}>
        <group rotation={[-Math.PI / 3, 0, 1]}>
          <Lightformer form="circle" intensity={4} rotation-x={Math.PI / 2} position={[0, 5, -9]} scale={2} />
          <Lightformer form="circle" intensity={2} rotation-y={Math.PI / 2} position={[-5, 1, -1]} scale={2} />
          <Lightformer form="circle" intensity={2} rotation-y={Math.PI / 2} position={[-5, -1, -1]} scale={2} />
          <Lightformer form="circle" intensity={2} rotation-y={-Math.PI / 2} position={[10, 1, 0]} scale={8} />
          <Lightformer form="ring" color="#a5b4fc" intensity={1} rotation-y={Math.PI / 2} position={[-0.1, -1, -5]} scale={10} />
        </group>
      </Environment>
    </>
  );
}

class ViewerErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : "Unknown error" };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-1 p-6 text-center text-sm text-slate-500">
          <p className="font-medium text-slate-700">The 3D preview could not be displayed.</p>
          <p>You can still download the .glb file. ({this.state.error.slice(0, 120)})</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function ModelViewer({ url, autoRotate, wireframe, resetSignal }: ModelViewerProps) {
  return (
    <ViewerErrorBoundary>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [1.6, 1.1, 1.9], fov: 40, near: 0.01, far: 100 }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
      >
        <StudioLights />
        <Suspense fallback={<Loader />}>
          <Bounds fit clip observe margin={1.25}>
            <Model url={url} wireframe={wireframe} />
            <Refit signal={resetSignal} />
          </Bounds>
          <ContactShadows position={[0, -0.002, 0]} scale={4} blur={2.4} opacity={0.45} far={2} resolution={512} />
        </Suspense>
        <OrbitControls makeDefault autoRotate={autoRotate} autoRotateSpeed={1.2} enableDamping minDistance={0.2} maxDistance={20} />
      </Canvas>
    </ViewerErrorBoundary>
  );
}

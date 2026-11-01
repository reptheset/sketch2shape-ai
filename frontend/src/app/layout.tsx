import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import type { ReactNode } from "react";
import "./globals.css";

// next/font downloads these once and serves them locally — no runtime Google requests.
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit", display: "swap" });

export const metadata: Metadata = {
  title: "Sketch2Shape AI — From a 2D sketch to a 3D asset",
  description: "Draw or upload a single-view sketch and turn it into a textured 3D model (.glb).",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${outfit.variable}`}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}

import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { Play, ArrowRight, X } from "lucide-react";
import demoVideo from "@assets/1764326196702_original-d776ac35-dd23-4f4c-9b3a-702fe541523c_1764326788637.mp4";
import hailuoBgVideo from "@assets/person.mp4";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import darkGradientBg from "@assets/Screenshot 2025-09-25 162657_1759214694717.png";
import CanvasVideo from "@/components/CanvasVideo";

const HeroSection = () => {
  const [, setLocation] = useLocation();
  const [showVideoDialog, setShowVideoDialog] = useState(false);

  const handleGetStarted = () => {
    setLocation("/coming-soon");
  };

  return (
    <section id="home" className="relative overflow-hidden h-screen flex items-center pt-[var(--nav-height)]">
      {/* Enhanced 3D Animated Background */}
      <div className="absolute inset-0">
        {/* Background video using canvas - no browser controls */}
        <CanvasVideo
          src={hailuoBgVideo}
          className="absolute inset-0 w-full h-full object-cover"
          poster={darkGradientBg}
        />
        {/* subtle dark tint overlay for readability */}
        <div className="absolute inset-0 bg-black/50 z-0 pointer-events-none" />

        {/* 3D Animated Grid Layers */}
        <motion.div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(6, 182, 212, 0.15) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(6, 182, 212, 0.15) 1px, transparent 1px)
            `,
            backgroundSize: "60px 60px",
            transform: "perspective(800px) rotateX(60deg) translateZ(-200px)",
          }}
          animate={{
            backgroundPosition: ["0px 0px", "60px 60px"],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Secondary diagonal grid */}
        <motion.div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `
              linear-gradient(45deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
              linear-gradient(-45deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: "40px 40px",
            transform: "perspective(600px) rotateX(50deg) translateZ(-150px)",
          }}
          animate={{
            backgroundPosition: ["0px 0px", "40px 40px"],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Floating particles for depth */}
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0.1, 0.6, 0.1],
              scale: [1, 1.8, 1],
              y: [0, -30, 0],
            }}
            transition={{
              duration: 3 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 3,
            }}
          />
        ))}

        {/* Animated gradient orbs for depth */}
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-br from-cyan-500/20 to-transparent rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            x: [0, 50, 0],
            y: [0, -30, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-br from-purple-500/20 to-transparent rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            x: [0, -40, 0],
            y: [0, 40, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full relative z-10">
        <div className="text-left space-y-6 max-w-2xl">
          {/* AI Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex justify-start"
          >
            <Badge variant="outline" className="glass-strong shadow-lg px-4 py-2 text-white border-white/20 bg-black/30">
              <div className="w-2 h-2 bg-primary rounded-full mr-2 animate-pulse" />
              <span className="font-sans tracking-wide">POWERED BY PATENTED SOLUTIONS</span>
            </Badge>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-semibold leading-tight tracking-wide text-white"
          >
            Complete Rack Audits in Minutes
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-white/85 text-lg max-w-2xl"
          >
            AI-powered rack documentation that automatically identifies racks, switches, patch panels, ports, and cables - turning hours of manual work into minutes of scanning
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 justify-start"
          >
            <Button
              size="default"
              className="group bg-gradient-to-r from-primary to-cyan-500 hover:from-primary/90 hover:to-cyan-600 text-white border-0 shadow-2xl hover:shadow-primary/50 transition-all duration-300"
              onClick={handleGetStarted}
              data-testid="button-get-started"
            >
              Get Started
              <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button
              variant="outline"
              size="default"
              className="group border-2 border-white/20 bg-black/10 hover:bg-black/20 hover:border-white/30 backdrop-blur-sm text-white"
              onClick={() => setShowVideoDialog(true)}
              data-testid="button-watch-demo"
            >
              Watch Demo
              <Play className="ml-2 w-4 h-4 group-hover:scale-110 transition-transform" />
            </Button>
          </motion.div>
        </div>
      </div>

      {/* Video Demo Dialog */}
      <Dialog open={showVideoDialog} onOpenChange={setShowVideoDialog}>
        <DialogContent className="max-w-5xl border border-white/8 rounded-lg p-3 sm:p-4 bg-slate-900/90">
          <div className="relative">
            <button
              type="button"
              aria-label="Close demo"
              className="absolute right-2 top-2 text-white/60 hover:text-white p-2 rounded-md transition-colors z-20"
              onClick={() => setShowVideoDialog(false)}
            >
              <X className="w-5 h-5" />
            </button>

            <div className="aspect-video rounded-md overflow-hidden bg-black/80 border border-white/6 shadow-lg">
              <video className="w-full h-full object-cover" controls autoPlay playsInline data-testid="video-demo">
                <source src={demoVideo} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-black/0 via-black/10 to-black/0" />
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
};

export default HeroSection;
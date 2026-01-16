import { motion } from "framer-motion";
import { useEffect, useState, useMemo } from "react";

interface SplashScreenProps {
  onComplete: () => void;
}

export default function SplashScreen({ onComplete }: SplashScreenProps) {
  const [phase, setPhase] = useState(0);

  const particles = useMemo(
    () =>
      Array.from({ length: 180 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 120,
        size: Math.random() * 2.5 + 0.4,
        duration: Math.random() * 3.5 + 2.5,
        delay: Math.random() * 1.8,
        color: ["#1e3a5f", "#2e5a8f", "#3a7cbf"][Math.floor(Math.random() * 3)],
      })),
    []
  );

  useEffect(() => {
    const phase1Timer = setTimeout(() => setPhase(1), 2800);
    const phase2Timer = setTimeout(() => setPhase(2), 4200);
    const completeTimer = setTimeout(() => onComplete(), 5000);

    return () => {
      clearTimeout(phase1Timer);
      clearTimeout(phase2Timer);
      clearTimeout(completeTimer);
    };
  }, [onComplete]);

  return (
    <motion.div
      className="fixed inset-0 z-[200] overflow-hidden"
      initial={{ opacity: 1 }}
      animate={{ opacity: phase === 2 ? 0 : 1 }}
      transition={{ duration: 0.9 }}
      style={{
        background: "linear-gradient(135deg, #0a0e27 0%, #1a1f3a 40%, #0f1729 70%, #000008 100%)",
        backdropFilter: "blur(1px)",
      }}
    >
      {/* Premium dark blue glow zones */}
      <motion.div
        className="absolute top-1/3 left-1/4 w-[600px] h-[600px] rounded-full blur-[130px] pointer-events-none"
        style={{
          background: "radial-gradient(circle, rgba(30, 58, 95, 0.15), transparent)",
        }}
        animate={{
          scale: [0.8, 1.2, 0.9],
          opacity: [0.2, 0.35, 0.2],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      <motion.div
        className="absolute bottom-1/4 right-1/3 w-[700px] h-[700px] rounded-full blur-[140px] pointer-events-none"
        style={{
          background: "radial-gradient(circle, rgba(58, 124, 191, 0.12), transparent)",
        }}
        animate={{
          scale: [1, 1.15, 0.95],
          opacity: [0.15, 0.3, 0.15],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          delay: 0.8,
          ease: "easeInOut",
        }}
      />

      {/* Central pulsing light sphere */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        animate={{
          opacity: [0.08, 0.2, 0.08],
        }}
        transition={{
          duration: 3.5,
          repeat: Infinity,
        }}
      >
        <div
          className="w-[450px] h-[450px] rounded-full blur-[110px]"
          style={{
            background: "radial-gradient(circle, rgba(58, 124, 191, 0.2), transparent)",
          }}
        />
      </motion.div>

      {/* Dynamic radial energy lines */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={`energy-line-${i}`}
            className="absolute"
            style={{
              width: "1.5px",
              height: "380px",
              background: `linear-gradient(to bottom, transparent, rgba(46, 90, 143, 0.35), transparent)`,
              transformOrigin: "center",
              transform: `rotate(${(i * 360) / 20}deg)`,
            }}
            animate={{
              opacity: [0.15, 0.5, 0.15],
              height: ["250px", "420px", "250px"],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              delay: i * 0.08,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      {/* Rising particles with dark blue glow */}
      {particles.map((p) => (
        <motion.div
          key={`particle-${p.id}`}
          className="absolute rounded-full"
          style={{
            width: p.size + "px",
            height: p.size + "px",
            left: `${p.x}%`,
            top: `${p.y}%`,
            background: p.color,
            boxShadow: `0 0 ${p.size * 5}px ${p.color}, 0 0 ${p.size * 8}px rgba(58, 124, 191, 0.3)`,
          }}
          animate={{
            y: [-50, -580],
            x: [0, (Math.random() - 0.5) * 100],
            opacity: [0, 0.8, 0],
            scale: [0.3, 1, 0.1],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "easeOut",
          }}
        />
      ))}

      {/* RACKTRACK - Rotating with dark blue tones */}
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          initial={{ scale: 0.15, opacity: 0, rotateZ: -50 }}
          animate={{
            scale: phase === 2 ? 0 : 1,
            opacity: phase === 2 ? 0 : 1,
            rotateZ: phase === 2 ? 50 : 0,
            rotateY: phase === 2 ? 100 : [0, 360],
          }}
          transition={{
            scale: { type: "spring", stiffness: 100, damping: 20, duration: 0.7 },
            opacity: { duration: 0.7 },
            rotateZ: { duration: 0.7 },
            rotateY: { duration: 6, repeat: Infinity, ease: "linear" },
          }}
          style={{
            transformStyle: "preserve-3d",
          }}
        >
          {/* Deep blur layer */}
          <motion.div
            className="absolute inset-0 text-8xl font-black text-center tracking-[0.12em]"
            style={{
              color: "rgba(30, 58, 95, 0.2)",
              filter: "blur(12px)",
              left: "-3px",
              top: "-2px",
            }}
            animate={{
              scale: [1, 1.08, 1],
            }}
            transition={{
              duration: 2.2,
              repeat: Infinity,
            }}
          >
            RACKTRACK
          </motion.div>

          {/* Medium blur layer */}
          <motion.div
            className="absolute inset-0 text-8xl font-black text-center tracking-[0.12em]"
            style={{
              background: "linear-gradient(135deg, #1e3a5f 0%, #3a7cbf 50%, #1e3a5f 100%)",
              backgroundSize: "200% auto",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              filter: "blur(4px)",
              opacity: 0.35,
            }}
            animate={{
              backgroundPosition: ["0% center", "100% center"],
            }}
            transition={{
              duration: 3.5,
              repeat: Infinity,
            }}
          >
            RACKTRACK
          </motion.div>

          {/* Main sharp text - deep blue gradient */}
          <motion.div
            className="text-8xl font-black text-center tracking-[0.12em] relative z-10"
            style={{
              background: "linear-gradient(135deg, #2e5a8f 0%, #5a9fd4 40%, #2e5a8f 70%, #1e3a5f 100%)",
              backgroundSize: "250% auto",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              filter: "drop-shadow(0 0 25px rgba(58, 124, 191, 0.6)) drop-shadow(0 0 50px rgba(46, 90, 143, 0.3))",
              textShadow: "0 0 30px rgba(58, 124, 191, 0.4)",
            }}
            animate={{
              backgroundPosition: ["0% center", "100% center"],
              scale: [1, 1.015, 1],
            }}
            transition={{
              backgroundPosition: { duration: 3.5, repeat: Infinity },
              scale: { duration: 1.8, repeat: Infinity, ease: "easeInOut" },
            }}
          >
            RACKTRACK
          </motion.div>

          {/* Subtle accent glow */}
          <motion.div
            className="absolute -inset-20 pointer-events-none"
            style={{
              borderRadius: "50%",
              boxShadow: "inset 0 0 60px rgba(58, 124, 191, 0.08), 0 0 80px rgba(46, 90, 143, 0.1)",
            }}
            animate={{
              opacity: [0.2, 0.4, 0.2],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
            }}
          />
        </motion.div>
      </div>

      {/* Bottom premium glow bar */}
      <motion.div
        className="absolute bottom-0 left-0 right-0 h-1"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(46, 90, 143, 0.6), rgba(58, 124, 191, 0.8), rgba(46, 90, 143, 0.6), transparent)",
          boxShadow: "0 0 50px rgba(58, 124, 191, 0.6), 0 -20px 50px rgba(46, 90, 143, 0.3)",
        }}
        animate={{
          scaleY: [1, 3, 1],
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          duration: 1.8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
    </motion.div>
  );
}

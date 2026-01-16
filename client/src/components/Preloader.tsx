import { motion } from "framer-motion";
import { useEffect, useState, useMemo, useRef } from "react";
import logoImage from "@assets/image_1761831098460-N9eO04bm_1764149372465.png";

interface PreloaderProps {
  isLoading: boolean;
}

export default function Preloader({ isLoading }: PreloaderProps) {
  const [displayPreloader, setDisplayPreloader] = useState(true);
  const [mousePosition, setMousePosition] = useState({ x: 50, y: 50 });
  const containerRef = useRef<HTMLDivElement>(null);

  const letters = "RACKTRACK".split("");

  const dataStreams = useMemo(
    () =>
      Array.from({ length: 25 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        delay: Math.random() * 3,
        duration: 2.5 + Math.random() * 3,
        width: Math.random() * 2 + 1,
      })),
    [],
  );

  const circuitNodes = useMemo(
    () =>
      Array.from({ length: 40 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 5 + 2,
        delay: Math.random() * 3,
      })),
    [],
  );

  const floatingRacks = useMemo(
    () =>
      Array.from({ length: 12 }, (_, i) => ({
        id: i,
        x: 5 + (i % 6) * 16,
        y: i < 6 ? 8 : 82,
        scale: 0.5 + Math.random() * 0.4,
        delay: Math.random() * 2,
      })),
    [],
  );

  const hexagons = useMemo(
    () =>
      Array.from({ length: 20 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 30 + Math.random() * 50,
        delay: Math.random() * 2,
        rotation: Math.random() * 360,
      })),
    [],
  );

  const particles = useMemo(
    () =>
      Array.from({ length: 100 }, (_, i) => ({
        id: i,
        angle: (i / 100) * Math.PI * 2,
        radius: 180 + Math.random() * 120,
        size: Math.random() * 2.5 + 0.5,
        duration: 4 + Math.random() * 3,
        delay: Math.random() * 0.8,
      })),
    [],
  );

  const orbits = useMemo(
    () =>
      Array.from({ length: 4 }, (_, i) => ({
        id: i,
        radius: 140 + i * 50,
        duration: 5 + i * 1.2,
        dots: 5 + i * 2,
        reverse: i % 2 === 1,
      })),
    [],
  );

  const binaryStreams = useMemo(
    () =>
      Array.from({ length: 15 }, (_, i) => ({
        id: i,
        x: 5 + Math.random() * 90,
        delay: Math.random() * 4,
        chars: Array.from({ length: 12 }, () => Math.random() > 0.5 ? "1" : "0").join(""),
      })),
    [],
  );

  const pulseRings = useMemo(
    () =>
      Array.from({ length: 5 }, (_, i) => ({
        id: i,
        delay: i * 0.8,
      })),
    [],
  );

  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isLoading) {
        setDisplayPreloader(false);
      }
    }, 3500);

    return () => clearTimeout(timer);
  }, [isLoading]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setMousePosition({ x, y });
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  if (!displayPreloader && !isLoading) return null;

  return (
    <motion.div
      ref={containerRef}
      className="fixed inset-0 z-[150] overflow-hidden flex items-center justify-center"
      initial={{ opacity: 1 }}
      animate={{ opacity: isLoading ? 1 : 0 }}
      transition={{ duration: 1 }}
      style={{
        background:
          "radial-gradient(ellipse at 50% 50%, #0d1a2d 0%, #0a0f1a 30%, #050810 60%, #020305 100%)",
        pointerEvents: isLoading ? "auto" : "none",
      }}
    >
      {/* Animated grid lines */}
      <div className="absolute inset-0 overflow-hidden opacity-15">
        <svg className="w-full h-full">
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(58, 124, 191, 0.4)" strokeWidth="0.5" />
            </pattern>
          </defs>
          <motion.rect
            width="200%"
            height="200%"
            fill="url(#grid)"
            animate={{
              x: [0, -50],
              y: [0, -50],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </svg>
      </div>

      {/* Binary code streams */}
      {binaryStreams.map((stream) => (
        <motion.div
          key={`binary-${stream.id}`}
          className="absolute text-xs pointer-events-none"
          style={{
            left: `${stream.x}%`,
            color: "rgba(58, 124, 191, 0.4)",
            textShadow: "0 0 10px rgba(58, 124, 191, 0.6)",
            writingMode: "vertical-rl",
          }}
          animate={{
            y: ["-100px", "110vh"],
            opacity: [0, 0.6, 0.6, 0],
          }}
          transition={{
            duration: 8,
            delay: stream.delay,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          {stream.chars}
        </motion.div>
      ))}

      {/* Hexagonal pattern overlay */}
      {hexagons.map((hex) => (
        <motion.div
          key={`hex-${hex.id}`}
          className="absolute pointer-events-none"
          style={{
            left: `${hex.x}%`,
            top: `${hex.y}%`,
            width: hex.size,
            height: hex.size * 0.866,
          }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{
            opacity: [0, 0.2, 0],
            scale: [0.5, 1.2, 0.5],
            rotate: [hex.rotation, hex.rotation + 60, hex.rotation],
          }}
          transition={{
            duration: 6,
            delay: hex.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <svg viewBox="0 0 100 86.6" className="w-full h-full">
            <polygon
              points="50,0 100,25 100,75 50,100 0,75 0,25"
              fill="none"
              stroke="rgba(58, 124, 191, 0.5)"
              strokeWidth="1"
            />
          </svg>
        </motion.div>
      ))}

      {/* Data stream lines - vertical */}
      {dataStreams.map((stream) => (
        <motion.div
          key={`stream-${stream.id}`}
          className="absolute h-full pointer-events-none"
          style={{
            left: `${stream.x}%`,
            width: stream.width,
          }}
        >
          <motion.div
            className="absolute w-full"
            style={{
              height: "80px",
              background: `linear-gradient(to bottom, transparent, rgba(58, 124, 191, 0.5), rgba(90, 200, 255, 0.7), rgba(58, 124, 191, 0.5), transparent)`,
              boxShadow: "0 0 15px rgba(58, 124, 191, 0.4)",
            }}
            animate={{
              y: ["-80px", "100vh"],
            }}
            transition={{
              duration: stream.duration,
              delay: stream.delay,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </motion.div>
      ))}

      {/* Floating server rack icons */}
      {floatingRacks.map((rack) => (
        <motion.div
          key={`rack-${rack.id}`}
          className="absolute pointer-events-none"
          style={{
            left: `${rack.x}%`,
            top: `${rack.y}%`,
            transform: `scale(${rack.scale})`,
          }}
          initial={{ opacity: 0, y: 20 }}
          animate={{
            opacity: [0.15, 0.35, 0.15],
            y: [0, -12, 0],
          }}
          transition={{
            duration: 4,
            delay: rack.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <svg width="40" height="60" viewBox="0 0 40 60">
            <rect x="4" y="4" width="32" height="52" rx="2" fill="none" stroke="rgba(58, 124, 191, 0.5)" strokeWidth="1" />
            <rect x="8" y="10" width="24" height="6" rx="1" fill="rgba(58, 124, 191, 0.2)" stroke="rgba(58, 124, 191, 0.3)" />
            <rect x="8" y="20" width="24" height="6" rx="1" fill="rgba(58, 124, 191, 0.2)" stroke="rgba(58, 124, 191, 0.3)" />
            <rect x="8" y="30" width="24" height="6" rx="1" fill="rgba(58, 124, 191, 0.2)" stroke="rgba(58, 124, 191, 0.3)" />
            <rect x="8" y="40" width="24" height="6" rx="1" fill="rgba(58, 124, 191, 0.2)" stroke="rgba(58, 124, 191, 0.3)" />
            <circle cx="28" cy="13" r="1.5" fill="rgba(0, 255, 150, 0.7)" />
            <circle cx="28" cy="23" r="1.5" fill="rgba(0, 255, 150, 0.7)" />
            <circle cx="28" cy="33" r="1.5" fill="rgba(58, 124, 191, 0.7)" />
            <circle cx="28" cy="43" r="1.5" fill="rgba(255, 200, 0, 0.7)" />
          </svg>
        </motion.div>
      ))}

      {/* Circuit nodes with connections */}
      {circuitNodes.map((node, i) => (
        <motion.div
          key={`node-${node.id}`}
          className="absolute pointer-events-none"
          style={{
            left: `${node.x}%`,
            top: `${node.y}%`,
          }}
        >
          <motion.div
            className="rounded-full"
            style={{
              width: node.size,
              height: node.size,
              background: "rgba(58, 124, 191, 0.5)",
              boxShadow: "0 0 12px rgba(58, 124, 191, 0.7), 0 0 25px rgba(90, 200, 255, 0.3)",
            }}
            animate={{
              scale: [1, 1.4, 1],
              opacity: [0.2, 0.7, 0.2],
            }}
            transition={{
              duration: 3,
              delay: node.delay,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </motion.div>
      ))}

      {/* Mouse-following spotlight */}
      <motion.div
        className="absolute pointer-events-none"
        style={{
          width: "500px",
          height: "500px",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(58, 124, 191, 0.12) 0%, transparent 60%)",
          filter: "blur(30px)",
        }}
        animate={{
          left: `${mousePosition.x}%`,
          top: `${mousePosition.y}%`,
          x: "-50%",
          y: "-50%",
        }}
        transition={{ type: "spring", damping: 30, stiffness: 100 }}
      />

      {/* Pulse rings expanding from center */}
      {pulseRings.map((ring) => (
        <motion.div
          key={`pulse-${ring.id}`}
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none"
          style={{
            border: "1px solid rgba(58, 124, 191, 0.4)",
          }}
          animate={{
            width: ["0px", "600px"],
            height: ["0px", "600px"],
            opacity: [0.8, 0],
          }}
          transition={{
            duration: 4,
            delay: ring.delay,
            repeat: Infinity,
            ease: "easeOut",
          }}
        />
      ))}

      {/* Core pulsing energy orbs */}
      {[0, 1, 2].map((i) => (
        <motion.div
          key={`orb-${i}`}
          className="absolute rounded-full"
          style={{
            width: "350px",
            height: "350px",
            left: "50%",
            top: "50%",
            marginLeft: "-175px",
            marginTop: "-175px",
            background: `radial-gradient(circle, rgba(58, 124, 191, ${0.18 - i * 0.04}), rgba(30, 80, 150, ${0.1 - i * 0.02}), transparent)`,
            filter: "blur(70px)",
          }}
          animate={{
            scale: [0.7 + i * 0.1, 1.3 + i * 0.1, 0.7 + i * 0.1],
            opacity: [0.5 - i * 0.05, 0.9 - i * 0.1, 0.5 - i * 0.05],
          }}
          transition={{
            duration: 4 + i * 0.8,
            repeat: Infinity,
            delay: i * 0.4,
            ease: "easeInOut",
          }}
        />
      ))}

      {/* Rotating orbital rings with energy dots */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        {orbits.map((orbit) => (
          <motion.div
            key={`orbit-${orbit.id}`}
            className="absolute"
            animate={{
              scale: [0.94, 1.06, 0.94],
              rotate: orbit.reverse ? -360 : 360,
            }}
            transition={{
              scale: {
                duration: 7 + orbit.id * 0.5,
                repeat: Infinity,
                ease: "easeInOut",
              },
              rotate: {
                duration: orbit.duration,
                repeat: Infinity,
                ease: "linear",
              },
            }}
            style={{
              width: orbit.radius * 2,
              height: orbit.radius * 2,
              borderRadius: "50%",
            }}
          >
            <motion.div
              style={{
                width: "100%",
                height: "100%",
                border: `1px solid rgba(58, 124, 191, ${0.3 - orbit.id * 0.05})`,
                borderRadius: "50%",
                boxShadow: `0 0 15px rgba(58, 124, 191, ${0.15 - orbit.id * 0.03})`,
              }}
              animate={{
                opacity: [0.5, 0.85, 0.5],
              }}
              transition={{
                duration: 4 + orbit.id * 0.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: orbit.id * 0.3,
              }}
            >
              {Array.from({ length: orbit.dots }).map((_, dotIndex) => (
                <motion.div
                  key={`dot-${orbit.id}-${dotIndex}`}
                  className="absolute rounded-full"
                  style={{
                    width: 5 - orbit.id * 0.4,
                    height: 5 - orbit.id * 0.4,
                    background: `rgba(90, 200, 255, ${0.85 - orbit.id * 0.1})`,
                    boxShadow: `0 0 12px rgba(90, 200, 255, 0.8), 0 0 25px rgba(58, 124, 191, 0.5)`,
                    left: "50%",
                    top: "0",
                    marginLeft: -2.5 + orbit.id * 0.2,
                    marginTop: -2.5 + orbit.id * 0.2,
                    transform: `rotate(${(dotIndex / orbit.dots) * 360}deg) translateY(${orbit.radius}px)`,
                    transformOrigin: `50% ${orbit.radius}px`,
                  }}
                  animate={{
                    opacity: [0.6, 1, 0.6],
                    scale: [1, 1.2, 1],
                  }}
                  transition={{
                    duration: 2 + Math.random(),
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: dotIndex * 0.1,
                  }}
                />
              ))}
            </motion.div>
          </motion.div>
        ))}
      </div>

      {/* Flowing energy particles */}
      {particles.map((p) => (
        <motion.div
          key={`particle-${p.id}`}
          className="absolute rounded-full pointer-events-none"
          style={{
            width: p.size,
            height: p.size,
            background: `rgba(90, 200, 255, 0.85)`,
            boxShadow: `0 0 ${p.size * 4}px rgba(90, 200, 255, 0.7), 0 0 ${p.size * 2}px rgba(58, 124, 191, 0.5)`,
            left: "50%",
            top: "50%",
            marginLeft: -p.size / 2,
            marginTop: -p.size / 2,
          }}
          animate={{
            x: [
              Math.cos(p.angle) * p.radius,
              Math.cos(p.angle + 0.6) * (p.radius + 60),
              Math.cos(p.angle + 1.2) * p.radius,
            ],
            y: [
              Math.sin(p.angle) * p.radius,
              Math.sin(p.angle + 0.6) * (p.radius + 60),
              Math.sin(p.angle + 1.2) * p.radius,
            ],
            opacity: [0.15, 0.85, 0.15],
            scale: [0.4, 1.1, 0.4],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "easeInOut",
          }}
        />
      ))}

      {/* Lens flare effect behind logo */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        style={{
          background: "radial-gradient(circle at 50% 50%, rgba(90, 200, 255, 0.2), rgba(58, 124, 191, 0.12), transparent 50%)",
        }}
        animate={{
          opacity: [0.4, 0.85, 0.4],
          scale: [0.85, 1.1, 0.85],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Electric arcs */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={`arc-${i}`}
          className="absolute pointer-events-none"
          style={{
            left: "50%",
            top: "50%",
            width: "250px",
            height: "1.5px",
            transformOrigin: "left center",
            transform: `rotate(${i * 45}deg)`,
          }}
        >
          <motion.div
            style={{
              width: "100%",
              height: "100%",
              background: "linear-gradient(90deg, rgba(90, 200, 255, 0.7), rgba(58, 124, 191, 0.3), transparent)",
              boxShadow: "0 0 8px rgba(90, 200, 255, 0.5)",
            }}
            animate={{
              opacity: [0, 0.8, 0],
              scaleX: [0, 1, 0],
            }}
            transition={{
              duration: 1.2,
              delay: i * 0.25,
              repeat: Infinity,
              ease: "easeOut",
            }}
          />
        </motion.div>
      ))}


      {/* Central logo and brand container */}
      <motion.div
        className="relative z-20 flex flex-col items-center justify-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Logo with glow */}
        <motion.div
          className="relative mb-6"
          initial={{ scale: 0, rotate: -180, opacity: 0 }}
          animate={{ scale: 1, rotate: 0, opacity: 1 }}
          transition={{
            duration: 0.8,
            type: "spring",
            stiffness: 120,
            damping: 15,
          }}
        >
          <motion.div
            className="absolute -inset-8 rounded-full"
            style={{
              background: "radial-gradient(circle, rgba(58, 124, 191, 0.4), transparent 70%)",
              filter: "blur(30px)",
            }}
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.5, 1, 0.5],
            }}
            transition={{
              duration: 2.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          
          <motion.img
            src={logoImage}
            alt="RackTrack Logo"
            className="relative z-10"
            style={{
              width: "90px",
              height: "90px",
              objectFit: "contain",
              filter: "invert(1) brightness(1.2) sepia(1) saturate(5) hue-rotate(180deg) drop-shadow(0 0 25px rgba(90, 200, 255, 0.8))",
            }}
            animate={{
              scale: [1, 1.1, 1],
              filter: [
                "invert(1) brightness(1.2) sepia(1) saturate(5) hue-rotate(180deg) drop-shadow(0 0 25px rgba(90, 200, 255, 0.8))",
                "invert(1) brightness(1.4) sepia(1) saturate(6) hue-rotate(180deg) drop-shadow(0 0 40px rgba(90, 200, 255, 1))",
                "invert(1) brightness(1.2) sepia(1) saturate(5) hue-rotate(180deg) drop-shadow(0 0 25px rgba(90, 200, 255, 0.8))",
              ],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />

          <motion.div
            className="absolute -inset-4"
            style={{
              borderRadius: "50%",
              border: "2px solid transparent",
              borderTopColor: "rgba(90, 200, 255, 0.8)",
              borderRightColor: "rgba(58, 124, 191, 0.4)",
            }}
            animate={{ rotate: 360 }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "linear",
            }}
          />
          
          <motion.div
            className="absolute -inset-6"
            style={{
              borderRadius: "50%",
              border: "1px solid transparent",
              borderBottomColor: "rgba(90, 200, 255, 0.5)",
              borderLeftColor: "rgba(58, 124, 191, 0.3)",
            }}
            animate={{ rotate: -360 }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        </motion.div>

        {/* Big RACKTRACK text - letter by letter reveal */}
        <div className="flex gap-1 justify-center items-center" style={{ whiteSpace: "nowrap" }}>
          {letters.map((letter, index) => (
              <motion.div
              key={`letter-${index}`}
              className="relative"
              style={{
                fontSize: "clamp(32px, 8vw, 140px)",
                fontWeight: 900,
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
              initial={{
                opacity: 0,
                scale: 0,
                rotateY: -90,
                x: -50,
              }}
              animate={{
                opacity: 1,
                scale: 1,
                rotateY: 0,
                x: 0,
              }}
              transition={{
                delay: 0.5 + index * 0.12,
                duration: 0.6,
                type: "spring",
                stiffness: 150,
                damping: 12,
              }}
            >
              {/* Glow layer behind letter */}
              <motion.div
                className="absolute inset-0 pointer-events-none"
                style={{
                  color: "rgba(90, 200, 255, 0.5)",
                  filter: "blur(18px)",
                  zIndex: -1,
                }}
                initial={{ opacity: 0 }}
                animate={{
                  opacity: [0, 0.8, 0.4],
                }}
                transition={{
                  delay: 0.5 + index * 0.12 + 0.3,
                  duration: 0.8,
                  times: [0, 0.5, 1],
                }}
              >
                {letter}
              </motion.div>

              {/* Flash effect on reveal */}
              <motion.div
                className="absolute inset-0 pointer-events-none"
                style={{
                  color: "rgba(255, 255, 255, 1)",
                  filter: "blur(8px)",
                  zIndex: 1,
                }}
                initial={{ opacity: 0, scale: 1.5 }}
                animate={{
                  opacity: [0, 1, 0],
                  scale: [1.5, 1, 1],
                }}
                transition={{
                  delay: 0.5 + index * 0.12,
                  duration: 0.4,
                  times: [0, 0.3, 1],
                }}
              >
                {letter}
              </motion.div>

              {/* Main letter */}
              <motion.span
                style={{
                  color: "#ffffff",
                  display: "block",
                  textShadow: "0 0 25px rgba(90, 200, 255, 0.7), 0 0 50px rgba(58, 124, 191, 0.5)",
                }}
                animate={{
                  textShadow: [
                    "0 0 25px rgba(90, 200, 255, 0.7), 0 0 50px rgba(58, 124, 191, 0.5)",
                    "0 0 40px rgba(90, 200, 255, 1), 0 0 80px rgba(58, 124, 191, 0.8)",
                    "0 0 25px rgba(90, 200, 255, 0.7), 0 0 50px rgba(58, 124, 191, 0.5)",
                  ],
                }}
                transition={{
                  textShadow: {
                    delay: 1.5 + index * 0.1,
                    duration: 2.5,
                    repeat: Infinity,
                  },
                }}
              >
                {letter}
              </motion.span>
            </motion.div>
          ))}
        </div>

        {/* Tagline */}
        <motion.div
          className="mt-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.6 }}
        >
          <motion.p
            className="text-xs sm:text-sm md:text-base lg:text-lg tracking-[0.25em] uppercase"
            style={{
              color: "rgba(150, 200, 255, 0.75)",
              textShadow: "0 0 15px rgba(58, 124, 191, 0.4)",
            }}
            animate={{
              opacity: [0.6, 1, 0.6],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            Infrastructure Intelligence
          </motion.p>
        </motion.div>

        {/* Loading indicator */}
        <motion.div
          className="mt-8 flex items-center gap-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.5 }}
        >
          {[0, 1, 2, 3, 4].map((i) => (
            <motion.div
              key={`loader-${i}`}
              className="w-2.5 h-2.5 rounded-full"
              style={{
                background: "rgba(90, 200, 255, 0.85)",
                boxShadow: "0 0 12px rgba(90, 200, 255, 0.7)",
              }}
              animate={{
                scale: [1, 1.4, 1],
                opacity: [0.4, 1, 0.4],
              }}
              transition={{
                duration: 0.9,
                delay: i * 0.12,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ))}
        </motion.div>
      </motion.div>

      {/* Corner decorative brackets */}
      {[
        { x: "0", y: "0", rotate: 0 },
        { x: "100%", y: "0", rotate: 90 },
        { x: "100%", y: "100%", rotate: 180 },
        { x: "0", y: "100%", rotate: 270 },
      ].map((corner, i) => (
        <motion.div
          key={`corner-decor-${i}`}
          className="absolute pointer-events-none"
          style={{
            left: corner.x === "0" ? "15px" : "auto",
            right: corner.x === "100%" ? "15px" : "auto",
            top: corner.y === "0" ? "15px" : "auto",
            bottom: corner.y === "100%" ? "15px" : "auto",
            transform: `rotate(${corner.rotate}deg)`,
          }}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.5, scale: 1 }}
          transition={{ delay: 0.4 + i * 0.1, duration: 0.4 }}
        >
          <svg width="50" height="50" viewBox="0 0 50 50">
            <motion.path
              d="M 0 25 L 0 0 L 25 0"
              fill="none"
              stroke="rgba(58, 124, 191, 0.5)"
              strokeWidth="1.5"
              animate={{
                stroke: [
                  "rgba(58, 124, 191, 0.3)",
                  "rgba(90, 200, 255, 0.7)",
                  "rgba(58, 124, 191, 0.3)",
                ],
              }}
              transition={{
                duration: 2,
                delay: i * 0.15,
                repeat: Infinity,
              }}
            />
            <motion.circle
              cx="0"
              cy="0"
              r="3"
              fill="rgba(90, 200, 255, 0.7)"
              animate={{
                r: [2, 4, 2],
                opacity: [0.4, 0.9, 0.4],
              }}
              transition={{
                duration: 1.5,
                delay: i * 0.15,
                repeat: Infinity,
              }}
            />
          </svg>
        </motion.div>
      ))}

      {/* Horizontal scanning line */}
      <motion.div
        className="absolute left-0 right-0 h-[1.5px] pointer-events-none"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(90, 200, 255, 0.7), transparent)",
          boxShadow: "0 0 25px rgba(90, 200, 255, 0.5), 0 0 50px rgba(58, 124, 191, 0.3)",
        }}
        animate={{
          top: ["0%", "100%", "0%"],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Bottom energy bar */}
      <motion.div
        className="absolute bottom-0 left-0 right-0 h-1"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(58, 124, 191, 0.4), rgba(90, 200, 255, 0.9), rgba(58, 124, 191, 0.4), transparent)",
          boxShadow: "0 0 40px rgba(90, 200, 255, 0.7), 0 -15px 40px rgba(58, 124, 191, 0.3)",
        }}
        animate={{
          scaleY: [1, 4, 1],
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          duration: 1.6,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Top energy bar */}
      <motion.div
        className="absolute top-0 left-0 right-0 h-1"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(58, 124, 191, 0.35), rgba(90, 200, 255, 0.75), rgba(58, 124, 191, 0.35), transparent)",
          boxShadow: "0 0 35px rgba(90, 200, 255, 0.5), 0 15px 35px rgba(58, 124, 191, 0.25)",
        }}
        animate={{
          scaleY: [1, 2.5, 1],
          opacity: [0.4, 0.85, 0.4],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.3,
        }}
      />
    </motion.div>
  );
}

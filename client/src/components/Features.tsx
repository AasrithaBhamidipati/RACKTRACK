import {
  Cable,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Cpu,
  Eye,
  FileText,
  Network,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import aboutUsBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";

const Features = () => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [flippedCards, setFlippedCards] = useState<Set<number>>(new Set());
  const [sweepingCards, setSweepingCards] = useState<Set<number>>(new Set());
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const checkScrollButtons = () => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 10);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 10);
    }
  };

  const features = [
    {
      icon: Cpu,
      title: "AI Component Recognition",
      description:
        "Automatically identifies and classifies all rack components - racks, switches, patch panels, ports, and cables. Detects rack structure, unit positions, and device layouts with high accuracy.",
      details: [
        "Identifies rack type and structure automatically",
        "Recognizes racks, switches, patch panels, ports, and cables",
        "Adapts to different lighting conditions and camera angles",
        "Assigns unique IDs to each detected component",
        "Achieves 95%+ detection accuracy in typical environments",
      ],
    },
    {
      icon: Cable,
      title: "Cable-to-Port Mapping",
      description:
        "Automatically maps cables from source to destination without manual tracing. Identifies cable types, classifies ports, and creates complete connection topology - even in high-density racks with overlapping cables.",
      details: [
        "Detects and classifies Ethernet, fiber, and power cables",
        "Maps cable connections end-to-end across devices",
        "Handles overlapping and occluded cable paths",
        "Identifies RJ-45, SFP, QSFP, and console ports",
        "Generates clear port-to-port connectivity diagrams",
      ],
    },
    {
      icon: Eye,
      title: "Optical Label Recognition",
      description:
        "Advanced OCR reads printed, handwritten, and faded labels directly from rack images. Extracts vendor names, serial numbers, model numbers, and custom tags - even from blurred or low-light photos.",
      details: [
        "Extracts serial numbers, vendor names, and rack IDs",
        "Works on blurred, low-light, or partially obscured images",
        "Supports multiple label formats and handwritten text",
        "Validates label consistency across cable endpoints",
        "Integrates extracted data into structured reports",
      ],
    },
    {
      icon: ShieldCheck,
      title: "Anomaly Detection",
      description:
        "Assigns confidence scores to every detection and automatically flags anomalies like missing labels, occluded cables, overlapping wires, or mismatched ports for technician review.",
      details: [
        "Highlights low-confidence detections for review",
        "Flags cable overlaps, disconnected cables, and unrecognized devices",
        "Learns from user feedback to improve accuracy",
        "Provides visual confidence indicators for each detection",
        "Ensures high data integrity in audit reports",
      ],
    },
    {
      icon: Zap,
      title: "AR Visualization",
      description:
        "Augmented Reality overlays display cable mappings and device information in real-time on your mobile screen. Instantly identify connections and component details during rack inspections.",
      details: [
        "Color-coded overlays for cables and connections",
        "Port labels and IDs displayed over live camera view",
        "Highlights detected cables and connection paths",
        "Shows component details and confidence scores",
        "Interactive visual experience for rack inspections",
      ],
    },
    {
      icon: FileText,
      title: "Automated Documentation",
      description:
        "Generates professional, audit-ready reports automatically. Exports rack layouts, device inventories, cable mappings, and anomaly summaries in PDF, Excel, or JSON formats.",
      details: [
        "Creates visual and structured reports for each rack",
        "Exports to PDF, Excel, or JSON formats instantly",
        "Includes timestamped images and anomaly highlights",
        "Easy integration with IT management and documentation systems",
        "Standardized format for audits and compliance",
      ],
    },
    {
      icon: RefreshCcw,
      title: "User Session Management",
      description:
        "Ensures each user maintains an independent, secure session with a dedicated history. All user data is fully isolated, preventing any sharing or exposure across accounts.",
      details: [
        "Independent, secure sessions for every user",
        "Dedicated history and activity tracking per account",
        "Full data isolation between users",
        "Prevents sharing or exposure of user data",
        "Supports compliance and privacy requirements",
      ],
    },
  ];

  // Check scroll position on mount and resize
  useEffect(() => {
    checkScrollButtons();
    window.addEventListener('resize', checkScrollButtons);
    return () => window.removeEventListener('resize', checkScrollButtons);
  }, []);

  // Navigation functions
  const scrollToLeft = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: -400, behavior: "smooth" });
      setTimeout(checkScrollButtons, 400);
    }
  };

  const scrollToRight = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: 400, behavior: "smooth" });
      setTimeout(checkScrollButtons, 400);
    }
  };

  const toggleFlip = (index: number) => {
    // Trigger sweep animation
    setSweepingCards((prev) => {
      const newSet = new Set(prev);
      newSet.add(index);
      return newSet;
    });

    // Remove sweep animation after 700ms (duration of animation)
    setTimeout(() => {
      setSweepingCards((prev) => {
        const newSet = new Set(prev);
        newSet.delete(index);
        return newSet;
      });
    }, 700);

    // Toggle flip state
    setFlippedCards((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <section id="features" className="relative overflow-hidden py-24">
      {/* Background */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${aboutUsBg})` }}
        />
        {/* Dark wash overlay for text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center px-6 py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg mb-6 shadow-2xl shadow-primary/30 relative overflow-hidden group">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/0 via-cyan-400/30 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <span className="relative text-white font-bold tracking-widest uppercase" style={{ fontSize: "14px", letterSpacing: "0.1em" }}>
              Features
            </span>
          </div>

          <h2 className="text-white mb-6 text-center">
            <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              The Intelligence Behind
            </span>{" "}
            RackTrack
          </h2>

          <span className="max-w-3xl mx-auto text-white/80">
            Explore the advanced technologies that transform manual audits into
            intelligent automation
          </span>
        </div>

        {/* Navigation Controls */}
        <div className="flex justify-center items-center gap-4 sm:gap-6 md:gap-8 mb-8 md:mb-12 px-4">
          <Button
            onClick={scrollToLeft}
            variant="outline"
            size="default"
            className={`rounded-full border-2 border-primary/30 bg-primary/5 hover:bg-primary/10 hover:border-primary/50 transition-all duration-300 w-9 h-9 p-0 ${
              canScrollLeft ? 'sm:flex opacity-100' : 'sm:flex opacity-0 pointer-events-none'
            } hidden`}
            data-testid="button-scroll-left"
            aria-label="Previous features"
          >
            <ChevronLeft className="w-4 h-4 text-primary" />
          </Button>

          <div className="text-center min-w-0">
            <h3 className="text-base sm:text-lg font-semibold text-white/90">
              Explore Our Features
            </h3>
            <p className="text-xs sm:text-sm text-white/60 mt-1">
              Click to see details
            </p>
          </div>

          <Button
            onClick={scrollToRight}
            variant="outline"
            size="default"
            className={`rounded-full border-2 border-primary/30 bg-primary/5 hover:bg-primary/10 hover:border-primary/50 transition-all duration-300 w-9 h-9 p-0 ${
              canScrollRight ? 'sm:flex opacity-100' : 'sm:flex opacity-0 pointer-events-none'
            } hidden`}
            data-testid="button-scroll-right"
            aria-label="Next features"
          >
            <ChevronRight className="w-4 h-4 text-primary" />
          </Button>
        </div>

        {/* Feature Cards Container */}
        <div className="relative px-4 sm:px-6 md:px-0">
          <div
            ref={scrollRef}
            className="flex gap-4 sm:gap-6 md:gap-8 overflow-x-auto pb-6 sm:pb-8 snap-x snap-mandatory"
            style={{
              scrollbarWidth: "none",
              msOverflowStyle: "none",
              WebkitOverflowScrolling: "touch",
            }}
            onScroll={checkScrollButtons}
            data-testid="features-container"
          >
            {features.map((feature, index) => {
              const isFlipped = flippedCards.has(index);
              const isSweeping = sweepingCards.has(index);

              return (
                <div
                  key={index}
                  className="flex-shrink-0 w-72 sm:w-80 md:w-96 snap-start"
                  data-testid={`feature-${feature.title.toLowerCase().replace(/\s+/g, "-")}`}
                >
                  <div
                    className="relative h-96 sm:h-[400px] md:h-[420px] w-full cursor-pointer group/card"
                    onClick={() => toggleFlip(index)}
                  >
                    {/* Front of card - Redesigned */}
                    <div
                      className={`absolute inset-0 rounded-3xl overflow-hidden shadow-2xl transition-opacity duration-700 ${
                        isFlipped
                          ? "opacity-0 pointer-events-none"
                          : "opacity-100"
                      }`}
                      style={{
                        background:
                          "linear-gradient(135deg, rgba(5, 5, 10, 0.95) 0%, rgba(10, 15, 30, 0.92) 50%, rgba(5, 5, 10, 0.95) 100%)",
                      }}
                    >
                      {/* Animated mesh gradient background */}
                      <div className="absolute inset-0 opacity-60">
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-violet-600/30 via-transparent to-transparent"></div>
                        <div className="absolute bottom-0 right-0 w-3/4 h-3/4 bg-gradient-to-tl from-blue-600/30 via-transparent to-transparent"></div>
                      </div>

                      {/* Floating orbs */}
                      <div className="absolute -top-16 -left-16 w-48 h-48 bg-gradient-to-br from-violet-500/40 to-purple-600/30 rounded-full blur-3xl opacity-50 group-hover/card:opacity-70 group-hover/card:scale-110 transition-all duration-700"></div>
                      <div className="absolute -bottom-20 -right-20 w-56 h-56 bg-gradient-to-tl from-blue-500/40 to-cyan-500/30 rounded-full blur-3xl opacity-50 group-hover/card:opacity-80 group-hover/card:scale-110 transition-all duration-700"></div>

                      {/* Grid pattern overlay */}
                      <div
                        className="absolute inset-0 opacity-[0.03]"
                        style={{
                          backgroundImage:
                            "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
                          backgroundSize: "50px 50px",
                        }}
                      ></div>

                      {/* Content */}
                      <div className="relative z-10 p-4 sm:p-5 md:p-6 h-full flex flex-col">
                        {/* Icon Container - Compact sizing */}
                        <div className="mb-3 sm:mb-4 relative">
                          <div className="absolute inset-0 bg-gradient-to-br from-violet-500/20 to-blue-500/20 rounded-2xl blur-lg"></div>
                          <div className="relative w-12 sm:w-14 h-12 sm:h-14 rounded-2xl bg-gradient-to-br from-violet-500/10 via-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-white/10 flex items-center justify-center group-hover/card:scale-110 group-hover/card:rotate-6 transition-all duration-500 shadow-xl">
                            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent"></div>
                            <feature.icon
                              className="relative w-6 sm:w-7 h-6 sm:h-7 text-white drop-shadow-lg"
                              strokeWidth={2}
                            />
                          </div>
                        </div>

                        {/* Text Content - Compact sizing for uniform alignment */}
                        <div className="flex-1 space-y-2 sm:space-y-3">
                          <h3 className="text-lg sm:text-xl font-bold text-white leading-tight tracking-tight">
                            {feature.title}
                          </h3>

                          <div className="h-0.5 w-16 bg-gradient-to-r from-violet-500/60 via-blue-500/40 to-transparent rounded-full"></div>

                          <p className="text-white/85 leading-relaxed text-sm sm:text-base">
                            {feature.description}
                          </p>
                        </div>

                        {/* Decorative element */}
                        <div className="mt-3 sm:mt-4 flex items-center gap-2 text-violet-400/60 text-xs sm:text-sm font-medium">
                          <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                          <span>Click to explore</span>
                        </div>
                      </div>

                      {/* Enhanced border with animated glow */}
                      <div className="absolute inset-0 rounded-3xl ring-1 ring-white/10 group-hover/card:ring-violet-400/40 transition-all duration-500"></div>
                      <div
                        className="absolute inset-0 rounded-3xl opacity-0 group-hover/card:opacity-100 transition-opacity duration-500"
                        style={{
                          boxShadow: "inset 0 0 60px rgba(139, 92, 246, 0.1)",
                        }}
                      ></div>

                      {/* Corner accents */}
                      <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-bl from-violet-500/20 via-violet-500/5 to-transparent opacity-50"></div>
                      <div className="absolute bottom-0 left-0 w-40 h-40 bg-gradient-to-tr from-blue-500/20 via-blue-500/5 to-transparent opacity-50"></div>

                      {/* Hover indicator - bottom right corner with ripple effect */}
                      <div className="absolute bottom-0 right-0 w-40 h-40 opacity-0 group-hover/card:opacity-100 transition-all duration-300 pointer-events-none">
                        {/* Animated ripple rings */}
                        <div className="absolute bottom-6 right-6 w-16 h-16 rounded-full border-2 border-violet-400/60 animate-ping"></div>
                        <div
                          className="absolute bottom-6 right-6 w-16 h-16 rounded-full border-2 border-blue-400/40 animate-ping"
                          style={{ animationDelay: "0.3s" }}
                        ></div>

                        {/* Central glow */}
                        <div className="absolute bottom-0 right-0 w-32 h-32">
                          <div className="absolute inset-0 bg-gradient-to-tl from-violet-500/70 via-blue-500/50 to-transparent rounded-br-3xl blur-2xl"></div>
                          <div className="absolute bottom-6 right-6 w-16 h-16 bg-gradient-to-tl from-violet-400 via-blue-400 to-transparent rounded-full blur-md opacity-80"></div>
                        </div>

                        {/* Sparkle hints */}
                        <div className="absolute bottom-8 right-8 w-2 h-2 bg-white rounded-full animate-pulse"></div>
                        <div
                          className="absolute bottom-12 right-10 w-1.5 h-1.5 bg-violet-300 rounded-full animate-pulse"
                          style={{ animationDelay: "0.5s" }}
                        ></div>
                        <div
                          className="absolute bottom-10 right-14 w-1 h-1 bg-blue-300 rounded-full animate-pulse"
                          style={{ animationDelay: "0.7s" }}
                        ></div>
                      </div>
                    </div>

                    {/* Back of card - Redesigned */}
                    <div
                      className={`absolute inset-0 rounded-3xl overflow-hidden shadow-2xl transition-opacity duration-700 ${
                        isFlipped
                          ? "opacity-100"
                          : "opacity-0 pointer-events-none"
                      }`}
                      style={{
                        background:
                          "linear-gradient(135deg, rgba(5, 8, 18, 1) 0%, rgba(10, 15, 28, 1) 50%, rgba(5, 8, 18, 1) 100%)",
                      }}
                    >
                      {/* Animated mesh gradient background */}
                      <div className="absolute inset-0 opacity-40">
                        <div className="absolute top-0 right-0 w-full h-full bg-gradient-to-bl from-blue-600/20 via-transparent to-transparent"></div>
                        <div className="absolute bottom-0 left-0 w-3/4 h-3/4 bg-gradient-to-tr from-violet-600/20 via-transparent to-transparent"></div>
                      </div>

                      {/* Floating orbs - reversed positions */}
                      <div className="absolute -top-16 -right-16 w-48 h-48 bg-gradient-to-bl from-blue-500/30 to-cyan-600/20 rounded-full blur-3xl opacity-40"></div>
                      <div className="absolute -bottom-20 -left-20 w-56 h-56 bg-gradient-to-tr from-violet-500/30 to-purple-500/20 rounded-full blur-3xl opacity-40"></div>

                      {/* Grid pattern overlay */}
                      <div
                        className="absolute inset-0 opacity-[0.03]"
                        style={{
                          backgroundImage:
                            "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
                          backgroundSize: "50px 50px",
                        }}
                      ></div>

                      {/* Content */}
                      <div className="relative z-10 p-3 sm:p-4 md:p-6 h-full flex flex-col">
                        {/* Header - Responsive design */}
                        <div className="mb-2 sm:mb-3 md:mb-4">
                          <div className="flex items-center gap-2 sm:gap-2.5 md:gap-3 mb-2 sm:mb-2.5 md:mb-3">
                            <div className="relative">
                              <div className="absolute inset-0 bg-gradient-to-br from-violet-500/20 to-blue-500/20 rounded-2xl blur-lg"></div>
                              <div className="relative w-9 sm:w-10 md:w-12 h-9 sm:h-10 md:h-12 rounded-2xl bg-gradient-to-br from-violet-500/10 via-blue-500/10 to-purple-500/10 backdrop-blur-xl border border-white/10 flex items-center justify-center shadow-xl">
                                <feature.icon
                                  className="w-4.5 h-4.5 sm:w-5 sm:h-5 md:w-6 md:h-6 text-white"
                                  strokeWidth={2}
                                />
                              </div>
                            </div>
                            <h3 className="text-sm sm:text-base md:text-lg font-bold text-foreground line-clamp-2">
                              <span className="bg-gradient-to-r from-white to-white/80 bg-clip-text text-transparent">
                                {feature.title}
                              </span>
                            </h3>
                          </div>
                          <div className="h-0.5 w-full bg-gradient-to-r from-violet-500/60 via-blue-500/40 to-transparent rounded-full"></div>
                        </div>

                        {/* Details list - No scrollbar, all content visible */}
                        <div className="flex-1 flex flex-col gap-1 sm:gap-1.5 md:gap-2">
                          {feature.details.map((detail, i) => (
                            <div
                              key={i}
                              className="flex items-start gap-1.5 sm:gap-2 md:gap-2.5 px-1 sm:px-1.5 md:px-2 py-0.5 sm:py-1 rounded-lg hover:bg-white/5 transition-all duration-300 group/item"
                            >
                              <div className="relative mt-0.5 flex-shrink-0">
                                <div className="absolute inset-0 bg-violet-500/20 rounded-full blur-sm group-hover/item:blur-md transition-all"></div>
                                <CheckCircle2
                                  className="relative w-3 h-3 sm:w-3.5 sm:h-3.5 text-violet-400 group-hover/item:text-blue-400 group-hover/item:scale-110 transition-all duration-300"
                                  strokeWidth={2}
                                />
                              </div>
                              <p className="text-muted-foreground leading-tight sm:leading-snug group-hover/item:text-white transition-colors text-[11px] sm:text-xs md:text-sm">
                                {detail}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Enhanced border with glow */}
                      <div className="absolute inset-0 rounded-3xl ring-1 ring-white/10"></div>
                      <div
                        className="absolute inset-0 rounded-3xl"
                        style={{
                          boxShadow: "inset 0 0 60px rgba(59, 130, 246, 0.1)",
                        }}
                      ></div>

                      {/* Corner accents */}
                      <div className="absolute top-0 left-0 w-40 h-40 bg-gradient-to-br from-violet-500/15 via-violet-500/5 to-transparent opacity-50"></div>
                      <div className="absolute bottom-0 right-0 w-40 h-40 bg-gradient-to-tl from-blue-500/15 via-blue-500/5 to-transparent opacity-50"></div>
                    </div>

                    {/* Sweep transition overlay - dynamic multi-layer wave animation */}
                    <div
                      className="absolute inset-0 rounded-3xl pointer-events-none overflow-hidden"
                      style={{
                        opacity: isSweeping ? 1 : 0,
                        transition: "opacity 0.15s ease-in",
                      }}
                    >
                      {/* Primary violet wave */}
                      <div
                        className="absolute inset-0"
                        style={{
                          background:
                            "radial-gradient(ellipse 140% 140% at 100% 100%, rgba(139, 92, 246, 1) 0%, rgba(124, 58, 237, 0.9) 25%, transparent 65%)",
                          transform: isSweeping
                            ? "translate(-5%, -5%) scale(3.5)"
                            : "translate(55%, 55%) scale(0.3)",
                          transition:
                            "transform 0.75s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                          transformOrigin: "bottom right",
                          filter: "blur(1px)",
                        }}
                      ></div>

                      {/* Secondary blue wave - slightly delayed and offset */}
                      <div
                        className="absolute inset-0"
                        style={{
                          background:
                            "radial-gradient(ellipse 130% 130% at 95% 95%, rgba(59, 130, 246, 0.9) 0%, rgba(37, 99, 235, 0.75) 30%, transparent 70%)",
                          transform: isSweeping
                            ? "translate(0%, 0%) scale(3.2)"
                            : "translate(60%, 60%) scale(0.4)",
                          transition:
                            "transform 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.08s",
                          transformOrigin: "bottom right",
                          mixBlendMode: "screen",
                        }}
                      ></div>

                      {/* Tertiary dark wave for depth */}
                      <div
                        className="absolute inset-0"
                        style={{
                          background:
                            "radial-gradient(ellipse 120% 120% at 90% 90%, rgba(31, 41, 55, 0.85) 0%, rgba(17, 24, 39, 0.6) 35%, transparent 75%)",
                          transform: isSweeping
                            ? "translate(5%, 5%) scale(3)"
                            : "translate(65%, 65%) scale(0.5)",
                          transition:
                            "transform 0.85s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.05s",
                          transformOrigin: "bottom right",
                        }}
                      ></div>

                      {/* Shimmer accent */}
                      <div
                        className="absolute inset-0"
                        style={{
                          background:
                            "linear-gradient(135deg, transparent 30%, rgba(255, 255, 255, 0.4) 50%, rgba(167, 139, 250, 0.3) 55%, transparent 70%)",
                          transform: isSweeping
                            ? "translate(-20%, -20%)"
                            : "translate(100%, 100%)",
                          transition:
                            "transform 0.65s cubic-bezier(0.4, 0, 0.2, 1) 0.15s",
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
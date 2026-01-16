import { motion } from "framer-motion";
import { Camera, Cpu, Tag, Cable, FileText, Archive, Search } from "lucide-react";
import aboutUsBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";
import logoImage from "@assets/Screenshot_2025-11-26_152113-removebg-preview_1764150705231.png";

type BorderPosition = 'left' | 'right' | 'bottom' | 'left-bottom' | 'right-bottom' | 'all';

const HowItWorks = () => {
  const features = [
    {
      title: "Scan Your Rack",
      description:
        "Capture high-resolution images or videos of rack fronts and backs using mobile, tablet, or webcam.",
      icon: Camera,
      gradient: "from-cyan-500 to-blue-500",
      glowColor: "rgba(6, 182, 212, 0.5)",
      borderPosition: 'left' as BorderPosition,
    },
    {
      title: "AI-Powered Detection",
      description:
        "Automatically detect all components - racks, switches, patch panels, ports, and cables.",
      icon: Cpu,
      gradient: "from-blue-500 to-indigo-500",
      glowColor: "rgba(59, 130, 246, 0.5)",
      borderPosition: 'right' as BorderPosition,
    },
    {
      title: "Labels Recognition",
      description:
        "Read all visible labels, serial numbers, and tags - even if blurred or partially hidden.",
      icon: Tag,
      gradient: "from-indigo-500 to-purple-500",
      glowColor: "rgba(99, 102, 241, 0.5)",
      borderPosition: 'left' as BorderPosition,
    },
    {
      title: "Auto Mapping",
      description:
        "Map cables to correct ports and analyze connections, highlighting issues automatically.",
      icon: Cable,
      gradient: "from-purple-500 to-pink-500",
      glowColor: "rgba(168, 85, 247, 0.5)",
      borderPosition: 'right' as BorderPosition,
    },
    {
      title: "Auto Reporting",
      description:
        "Automatically generate audit-ready reports in PDF, Excel, or JSON formats.",
      icon: FileText,
      gradient: "from-pink-500 to-rose-500",
      glowColor: "rgba(236, 72, 153, 0.5)",
      borderPosition: 'left-bottom' as BorderPosition,
    },
    {
      title: "Session Archiving",
      description:
        "Securely store scans and reports for each user session with historical data access.",
      icon: Archive,
      gradient: "from-orange-500 to-amber-500",
      glowColor: "rgba(249, 115, 22, 0.5)",
      borderPosition: 'bottom' as BorderPosition,
    },
    {
      title: "Smart Search",
      description:
        "Instantly find any device, cable, or connection across your entire infrastructure.",
      icon: Search,
      gradient: "from-emerald-500 to-teal-500",
      glowColor: "rgba(16, 185, 129, 0.5)",
      borderPosition: 'right-bottom' as BorderPosition,
    },
  ];

  const getBorderStyles = (position: BorderPosition, gradient: string, glowColor: string) => {
    const baseStyle = {
      position: 'absolute' as const,
      opacity: 0.4,
      transition: 'opacity 0.5s',
    };

    switch (position) {
      case 'left':
        return {
          ...baseStyle,
          left: 0,
          top: 0,
          bottom: 0,
          width: '2px',
          background: `linear-gradient(to bottom, ${glowColor.replace('0.5', '0.8')}, ${glowColor})`,
          boxShadow: `0 0 15px ${glowColor}`,
        };
      case 'right':
        return {
          ...baseStyle,
          right: 0,
          top: 0,
          bottom: 0,
          width: '2px',
          background: `linear-gradient(to bottom, ${glowColor.replace('0.5', '0.8')}, ${glowColor})`,
          boxShadow: `0 0 15px ${glowColor}`,
        };
      case 'bottom':
        return {
          ...baseStyle,
          bottom: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: `linear-gradient(to right, ${glowColor}, ${glowColor.replace('0.5', '0.8')}, ${glowColor})`,
          boxShadow: `0 0 15px ${glowColor}`,
        };
      case 'left-bottom':
        return null; // Will render two elements
      case 'right-bottom':
        return null; // Will render two elements
      default:
        return baseStyle;
    }
  };

  const FeatureCard = ({ feature, index, className = "" }: { feature: typeof features[0], index: number, className?: string }) => {
    const Icon = feature.icon;
    const borderPosition = feature.borderPosition;

    return (
      <motion.div
        data-testid={`step-${index}`}
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay: 0.1 + index * 0.1 }}
        whileHover={{
          scale: 1.02,
          y: -4,
        }}
        className={`group ${className}`}
        style={{
          perspective: "1000px",
        }}
      >
        <div
          className="relative h-full rounded-xl overflow-visible transition-all duration-500"
          style={{
            transformStyle: "preserve-3d",
          }}
        >
          {/* Outer glow effect - always visible, intensifies on hover */}
          <div
            className="absolute -inset-[1px] rounded-xl transition-all duration-500 group-hover:blur-md"
            style={{
              background: `linear-gradient(135deg, ${feature.glowColor}, rgba(139, 92, 246, 0.3), ${feature.glowColor})`,
              opacity: 0.4,
            }}
          />
          <div
            className="absolute -inset-[1px] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl"
            style={{
              background: `linear-gradient(135deg, ${feature.glowColor}, rgba(139, 92, 246, 0.5), ${feature.glowColor})`,
            }}
          />

          {/* Main card with 3D depth */}
          <div
            className="relative h-full p-6 rounded-xl transition-all duration-500"
            style={{
              background: "linear-gradient(145deg, rgba(15, 15, 20, 0.95), rgba(5, 5, 10, 0.98))",
              boxShadow: `
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05),
                inset 0 -1px 0 0 rgba(0, 0, 0, 0.3),
                0 10px 40px -10px rgba(0, 0, 0, 0.5),
                0 2px 10px -2px rgba(0, 0, 0, 0.3)
              `,
              border: "1px solid rgba(255, 255, 255, 0.08)",
            }}
          >
            {/* Glowing border overlay */}
            <div
              className="absolute inset-0 rounded-xl opacity-50 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
              style={{
                background: `linear-gradient(135deg, transparent 40%, ${feature.glowColor.replace('0.5', '0.1')} 50%, transparent 60%)`,
              }}
            />

            {/* Top highlight for 3D effect */}
            <div className="absolute top-0 left-4 right-4 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent" />

            {/* Content */}
            <div className="relative z-10">
              {/* Icon container */}
              <div className="relative mb-6">
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                  style={{
                    background: `linear-gradient(135deg, ${feature.glowColor.replace('0.5', '0.2')}, ${feature.glowColor.replace('0.5', '0.1')})`,
                    border: `1px solid ${feature.glowColor.replace('0.5', '0.3')}`,
                    boxShadow: `0 0 20px ${feature.glowColor.replace('0.5', '0.2')}`,
                  }}
                >
                  <Icon className="w-5 h-5 text-white" />
                </div>
                {/* Icon glow */}
                <div
                  className="absolute inset-0 w-11 h-11 rounded-xl opacity-0 group-hover:opacity-60 blur-xl transition-opacity duration-500"
                  style={{
                    background: `linear-gradient(135deg, ${feature.glowColor}, ${feature.glowColor})`,
                  }}
                />
              </div>

              <h3
                className="text-lg font-bold text-white mb-3 tracking-tight"
                data-testid={`title-${index}`}
              >
                {feature.title}
              </h3>
              <p
                className="text-white/70 text-sm leading-[1.8] group-hover:text-white/80 transition-colors duration-300"
                data-testid={`description-${index}`}
              >
                {feature.description}
              </p>
            </div>

            {/* Position-based gradient accents */}
            {borderPosition === 'left' && (
              <div
                className={`absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
              />
            )}
            {borderPosition === 'right' && (
              <div
                className={`absolute right-0 top-0 bottom-0 w-[2px] bg-gradient-to-b ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
              />
            )}
            {borderPosition === 'bottom' && (
              <div
                className={`absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
              />
            )}
            {borderPosition === 'left-bottom' && (
              <>
                <div
                  className={`absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                  style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
                />
                <div
                  className={`absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                  style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
                />
              </>
            )}
            {borderPosition === 'right-bottom' && (
              <>
                <div
                  className={`absolute right-0 top-0 bottom-0 w-[2px] bg-gradient-to-b ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                  style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
                />
                <div
                  className={`absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r ${feature.gradient} opacity-40 group-hover:opacity-90 transition-opacity duration-500`}
                  style={{ boxShadow: `0 0 15px ${feature.glowColor}` }}
                />
              </>
            )}
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <section id="how-it-works" className="relative overflow-hidden py-24">
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${aboutUsBg})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/95 via-black/90 to-black/95" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8 sm:mb-12 md:mb-16">
          <motion.div
            className="inline-flex items-center px-4 sm:px-6 py-2 sm:py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg mb-6 sm:mb-8 shadow-2xl shadow-primary/30"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <span className="text-white font-bold tracking-widest uppercase text-xs sm:text-sm">
              How It Works
            </span>
          </motion.div>

          <motion.h2
            className="text-white mb-4 sm:mb-6 text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              From Image
            </span>{" "}
            to Insight
          </motion.h2>

          <motion.p
            className="max-w-3xl mx-auto text-white/60 text-sm sm:text-base md:text-lg"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            Scan, detect, read, map, validate, and report - all automated. From
            image capture to audit-ready documentation in minutes.
          </motion.p>
        </div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5">
          {/* Left Column - Top Card */}
          <FeatureCard feature={features[0]} index={0} />

          {/* Center Card - Tall with Device Mockup */}
          <motion.div
            data-testid="step-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="md:row-span-2"
          >
            <div className="relative h-full min-h-[420px] md:min-h-full rounded-xl overflow-visible">
              {/* Outer glow */}
              <div className="absolute -inset-[1px] rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-cyan-500 blur-sm" />
              <div className="absolute -inset-2 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-cyan-500 blur-xl" />

              {/* Main card */}
              <div
                className="relative h-full p-6 rounded-xl overflow-hidden"
                style={{
                  background: "linear-gradient(145deg, rgba(15, 15, 20, 0.95), rgba(5, 5, 10, 0.98))",
                  boxShadow: `
                    inset 0 1px 0 0 rgba(255, 255, 255, 0.05),
                    inset 0 -1px 0 0 rgba(0, 0, 0, 0.3),
                    0 20px 60px -15px rgba(0, 0, 0, 0.5)
                  `,
                  border: "1px solid rgba(139, 92, 246, 0.3)",
                }}
              >
                {/* All sides gradient borders */}
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-70"
                  style={{ boxShadow: '0 0 15px rgba(139, 92, 246, 0.5)' }} />
                <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 opacity-70"
                  style={{ boxShadow: '0 0 15px rgba(139, 92, 246, 0.5)' }} />
                <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500 opacity-70"
                  style={{ boxShadow: '0 0 15px rgba(139, 92, 246, 0.5)' }} />
                <div className="absolute right-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-pink-500 via-purple-500 to-blue-500 opacity-70"
                  style={{ boxShadow: '0 0 15px rgba(139, 92, 246, 0.5)' }} />

                {/* Top highlight */}
                <div className="absolute top-0 left-8 right-8 h-[1px] bg-gradient-to-r from-transparent via-purple-400/30 to-transparent" />

                {/* Content */}
                <div className="relative z-10 h-full flex flex-col items-center justify-center">
                  {/* Rainbow gradient border device mockup */}
                  <div className="relative">
                    {/* Gradient glow behind device */}
                    <div className="absolute inset-0 blur-3xl opacity-50">
                      <div className="w-full h-full bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-3xl" />
                    </div>

                    {/* Device frame with gradient border */}
                    <div className="relative p-[3px] rounded-[2.5rem] bg-gradient-to-br from-blue-400 via-purple-500 via-pink-500 via-orange-400 to-yellow-400">
                      <div className="bg-black rounded-[2.2rem] p-4 w-48 h-80 md:w-52 md:h-96 flex flex-col items-center">
                        {/* Notch */}
                        <div className="w-20 h-6 bg-black rounded-full border border-white/10 mb-4 flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-white/20" />
                        </div>

                        {/* Screen content with RackTrack Logo */}
                        <div className="flex-1 w-full rounded-xl bg-gradient-to-b from-slate-900 to-slate-950 border border-white/5 flex flex-col items-center justify-center p-4">
                          {/* RackTrack Logo */}
                          <div className="relative mb-4">
                            <img
                              src={logoImage}
                              alt="RackTrack Logo"
                              className="w-20 h-20 object-contain relative z-10"
                            />
                            {/* Glow effect behind logo */}
                            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-500 opacity-40 blur-xl rounded-full" />
                          </div>
                          <div className="text-center">
                            <h4 className="text-white font-bold text-lg mb-1">RackTrack AI</h4>
                            <p className="text-white/50 text-xs leading-relaxed">Intelligent Infrastructure</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Caption below device */}
                  <div className="mt-6 text-center">
                    <h3 className="text-xl font-bold text-white mb-2">
                      Complete Automation
                    </h3>
                    <p className="text-white/60 text-sm max-w-xs leading-relaxed">
                      From scan to report in minutes, not hours
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right Column - Top Card */}
          <FeatureCard feature={features[1]} index={1} />

          {/* Left Column - Bottom Card */}
          <FeatureCard feature={features[2]} index={2} />

          {/* Right Column - Bottom Card */}
          <FeatureCard feature={features[3]} index={3} />

          {/* Bottom Row - Three Cards */}
          <FeatureCard feature={features[4]} index={4} />
          <FeatureCard feature={features[5]} index={5} />
          <FeatureCard feature={features[6]} index={6} />
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
import { motion } from "framer-motion";
import serverRackVideo from "@assets/fotor-video_remover_object-preview-20251104153726 1_1762252516621.mp4";
import aboutUsBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";
import datacenterBg from "@assets/generated_images/realistic_data_center_server_racks.png";
import CanvasVideo from "@/components/CanvasVideo";


const WhyChooseUs = () => {
  const coreFeatures = [
    {
      title: "Automated Audits",
      description: "AI-powered rack scanning and cable mapping completed in minutes",
      accentColor: "rgb(6, 182, 212)",
    },
    {
      title: "Precision Intelligence",
      description: "Accurate detection of every port, device, and cable using computer vision",
      accentColor: "rgb(34, 211, 238)",
    },
    {
      title: "End-to-End Visibility",
      description: "Gain a unified, comprehensive view of your entire data center infrastructure",
      accentColor: "rgb(8, 145, 178)",
    },
    {
      title: "Instant Reports",
      description: "Produce comprehensive, audit-ready reports optimized for effortless integration",
      accentColor: "rgb(14, 165, 233)",
    }
  ];

  // Renaming features to coreFeatures to match the original code's data structure
  const features = coreFeatures;

  return (
    <section className="relative overflow-hidden py-24">
      {/* Background */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${aboutUsBg})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Main Content Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-10 items-center">

          {/* Left side - Video */}
          <div className="order-2 lg:order-1 flex items-stretch">
            <div className="relative w-full flex flex-col min-h-[550px] lg:min-h-[650px]">
              <div className="relative rounded-2xl overflow-hidden bg-black group/video flex-1 border border-white/10 hover:border-cyan-400/30 transition-all duration-300 shadow-2xl">
                <CanvasVideo
                  src={serverRackVideo}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 border-2 border-transparent group-hover/video:border-cyan-400/40 rounded-2xl transition-colors duration-300 pointer-events-none"></div>
              </div>
            </div>
          </div>

          {/* Right side - Content */}
          <div className="order-1 lg:order-2 flex flex-col h-full">

            {/* Header Badge */}
            <div className="mb-6 flex justify-center lg:justify-start">
              <div className="inline-flex items-center px-6 py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg shadow-2xl shadow-primary/30 relative overflow-hidden group">
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/0 via-cyan-400/30 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative text-white font-bold tracking-widest uppercase" style={{ fontSize: "14px", letterSpacing: "0.1em" }}>
                  What Sets Us Apart
                </span>
              </div>
            </div>

            {/* Main Heading */}
            <h2 className="text-white mb-4 text-left lg:text-left text-2xl md:text-3xl lg:text-3xl font-bold leading-tight">
              <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Smarter infrastructure management
              </span>{" "}
              <span className="text-white">through AI automation</span>
            </h2>

            {/* Subtitle */}
            <p className="mb-8 text-left lg:text-left text-white/80 text-base leading-relaxed max-w-lg">
              RackTrack delivers instant, accurate, and scalable insights across every rack and network component
            </p>

            {/* Feature Cards Grid */}
            <div className="grid md:grid-cols-2 gap-4">
              {features.map((feature, index) => (
                <motion.div
                  key={index}
                  className="group relative cursor-pointer"
                  data-testid={`card-feature-${index}`}
                  initial={{ opacity: 0, y: 25, scale: 0.95 }}
                  whileInView={{ opacity: 1, y: 0, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
                  whileHover={{ y: -6 }}
                >
                  {/* Outer glow bloom on hover */}
                  <div 
                    className="absolute -inset-1 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl pointer-events-none"
                    style={{
                      background: `radial-gradient(ellipse at 50% 50%, ${feature.accentColor}30, transparent 70%)`,
                    }}
                  />

                  {/* Main card */}
                  <div 
                    className="relative rounded-xl overflow-hidden min-h-[180px] transition-all duration-300"
                    style={{
                      background: 'linear-gradient(145deg, rgba(0, 20, 45, 0.9) 0%, rgba(0, 8, 20, 0.95) 50%, rgba(2, 15, 35, 0.9) 100%)',
                      boxShadow: '0 10px 40px rgba(0, 0, 0, 0.7)',
                    }}
                  >
                    {/* Gradient border overlay - dark blue, cyan, black, white */}
                    <div 
                      className="absolute inset-0 rounded-xl pointer-events-none"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0, 50, 100, 0.7) 0%, rgba(6, 182, 212, 0.4) 25%, rgba(255, 255, 255, 0.15) 50%, rgba(6, 182, 212, 0.3) 75%, rgba(0, 20, 50, 0.6) 100%)',
                        padding: '1px',
                        mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                        maskComposite: 'xor',
                        WebkitMaskComposite: 'xor',
                      }}
                    />

                    {/* Noise texture overlay */}
                    <div 
                      className="absolute inset-0 opacity-[0.02] pointer-events-none"
                      style={{
                        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
                      }}
                    />

                    {/* Inner border glow on hover - cyan theme */}
                    <div 
                      className="absolute inset-0 rounded-xl pointer-events-none opacity-0 group-hover:opacity-100 transition-all duration-300"
                      style={{
                        boxShadow: 'inset 0 0 35px rgba(6, 182, 212, 0.08), 0 0 30px rgba(6, 182, 212, 0.1)',
                      }}
                    />

                    {/* Corner tech chips - cyan theme */}
                    <div 
                      className="absolute top-4 left-4 w-2 h-2 rounded-sm opacity-40 group-hover:opacity-80 transition-opacity duration-300"
                      style={{ background: 'rgb(6, 182, 212)' }}
                    />
                    <div 
                      className="absolute top-4 left-7 w-1 h-1 rounded-sm opacity-25 group-hover:opacity-60 transition-opacity duration-300"
                      style={{ background: 'rgba(255, 255, 255, 0.8)' }}
                    />
                    <div 
                      className="absolute bottom-4 right-4 w-2 h-2 rounded-sm opacity-40 group-hover:opacity-80 transition-opacity duration-300"
                      style={{ background: 'rgb(6, 182, 212)' }}
                    />

                    {/* Animated scan line - cyan theme */}
                    <motion.div
                      className="absolute left-0 right-0 h-[1px] opacity-0 group-hover:opacity-80 pointer-events-none z-20"
                      style={{
                        background: 'linear-gradient(90deg, transparent 0%, rgb(6, 182, 212) 50%, transparent 100%)',
                        boxShadow: '0 0 12px rgba(6, 182, 212, 0.6)',
                      }}
                      initial={{ top: 0 }}
                      animate={{ top: ['0%', '100%'] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        ease: 'linear',
                      }}
                    />

                    {/* Radial core glow - cyan theme */}
                    <div 
                      className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                      style={{
                        background: 'radial-gradient(ellipse at 80% 80%, rgba(6, 182, 212, 0.08), transparent 60%)',
                      }}
                    />

                    {/* Content */}
                    <div className="relative z-10 p-6 h-full flex flex-col items-center justify-center text-center min-h-[180px]">
                      {/* Title - centered by default, moves up on hover */}
                      <div className="transition-all duration-400 ease-out group-hover:-translate-y-3">
                        <h3 
                          className="text-lg font-bold mb-3 transition-all duration-300 text-white"
                          style={{ 
                            textShadow: '0 0 30px rgba(6, 182, 212, 0.3)',
                          }}
                        >
                          {feature.title}
                        </h3>
                        {/* Accent underline - cyan gradient */}
                        <div 
                          className="h-[2px] w-12 mx-auto group-hover:w-20 transition-all duration-300 rounded-full"
                          style={{ 
                            background: 'linear-gradient(90deg, rgba(6, 182, 212, 0.4), rgba(6, 182, 212, 1), rgba(255, 255, 255, 0.6), rgba(6, 182, 212, 1), rgba(6, 182, 212, 0.4))',
                            boxShadow: '0 0 12px rgba(6, 182, 212, 0.5)',
                          }}
                        />
                      </div>

                      {/* Description - reveals on hover */}
                      <div className="overflow-hidden max-h-0 group-hover:max-h-28 opacity-0 group-hover:opacity-100 translate-y-3 group-hover:translate-y-0 transition-all duration-400 ease-out mt-0 group-hover:mt-4">
                        <p className="text-sm leading-relaxed text-white/90 transition-colors duration-300">
                          {feature.description}
                        </p>
                      </div>
                    </div>

                    {/* Top edge highlight - cyan/white gradient */}
                    <div 
                      className="absolute top-0 left-0 right-0 h-[1px] opacity-50"
                      style={{
                        background: 'linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.6), rgba(255, 255, 255, 0.3), rgba(6, 182, 212, 0.6), transparent)',
                      }}
                    />

                    {/* Floating particle effects - cyan theme */}
                    <motion.div
                      className="absolute w-1 h-1 rounded-full opacity-0 group-hover:opacity-70 pointer-events-none"
                      style={{
                        background: 'rgb(6, 182, 212)',
                        boxShadow: '0 0 8px rgba(6, 182, 212, 0.8)',
                        right: '20%',
                        top: '30%',
                      }}
                      animate={{
                        y: [0, -10, 0],
                        opacity: [0.3, 0.7, 0.3],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: 'easeInOut',
                      }}
                    />
                    <motion.div
                      className="absolute w-1.5 h-1.5 rounded-full opacity-0 group-hover:opacity-60 pointer-events-none"
                      style={{
                        background: 'rgba(255, 255, 255, 0.9)',
                        boxShadow: '0 0 10px rgba(255, 255, 255, 0.6)',
                        right: '10%',
                        bottom: '40%',
                      }}
                      animate={{
                        y: [0, -15, 0],
                        opacity: [0.2, 0.5, 0.2],
                      }}
                      transition={{
                        duration: 4,
                        repeat: Infinity,
                        ease: 'easeInOut',
                        delay: 0.5,
                      }}
                    />
                  </div>
                </motion.div>
              ))}</div>
          </div>
        </div>
      </div>

    </section>
  );
};

export default WhyChooseUs;
import { motion } from "framer-motion";
import rackImage1 from "@assets/Complete Rack View_1763386913885.png";
import rackImage2 from "@assets/Device Recognition.png";
import rackImage3 from "@assets/Cable Mapping.png";
import rackImage4 from "@assets/Switching Intelligence_1762256088044.jpg";
import rackImage5 from "@assets/Port-Level Precision.png";
import rackImage6 from "@assets/Audit Reports.png";
import aboutUsBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";

const Gallery = () => {
  const galleryItems = [
    { 
      id: 1, 
      image: rackImage1, 
      category: "overview",
      title: "Complete Rack View",
      description: "Full infrastructure visibility with AI-powered component detection",
      gradient: "from-cyan-500 to-blue-500",
      glowColor: "rgba(6, 182, 212, 0.6)"
    },
    { 
      id: 2, 
      image: rackImage2, 
      category: "devices",
      title: "Device Recognition",
      description: "Automatic identification of all hardware components and models",
      gradient: "from-blue-500 to-indigo-500",
      glowColor: "rgba(59, 130, 246, 0.6)"
    },
    { 
      id: 3, 
      image: rackImage3, 
      category: "cabling",
      title: "Cable Mapping",
      description: "Intelligent cable tracing and port-to-port connection analysis",
      gradient: "from-indigo-500 to-purple-500",
      glowColor: "rgba(99, 102, 241, 0.6)"
    },
    { 
      id: 4, 
      image: rackImage4, 
      category: "visualization",
      title: "Structured Cabling",
      description: "Visual representation of your entire cabling infrastructure",
      gradient: "from-purple-500 to-pink-500",
      glowColor: "rgba(168, 85, 247, 0.6)"
    },
    { 
      id: 5, 
      image: rackImage5, 
      category: "precision",
      title: "Port-Level Precision",
      description: "Detailed port mapping with connection status indicators",
      gradient: "from-orange-500 to-amber-500",
      glowColor: "rgba(249, 115, 22, 0.6)"
    },
    { 
      id: 6, 
      image: rackImage6, 
      category: "reports",
      title: "Audit Reports",
      description: "Comprehensive documentation ready for compliance audits",
      gradient: "from-emerald-500 to-teal-500",
      glowColor: "rgba(16, 185, 129, 0.6)"
    }
  ];

  // Triple the items for seamless infinite scroll
  const scrollItems = [...galleryItems, ...galleryItems, ...galleryItems];

  return (
    <section id="gallery" className="relative overflow-hidden py-24">
      <div className="absolute inset-0">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${aboutUsBg})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <motion.div 
            className="inline-flex items-center px-6 py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg mb-6 shadow-2xl shadow-primary/30 relative overflow-hidden group"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/0 via-cyan-400/30 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <span className="relative text-white font-bold tracking-widest uppercase" style={{ fontSize: "14px", letterSpacing: "0.1em" }}>
              Gallery
            </span>
          </motion.div>

          <motion.h2 
            className="text-white mb-6 text-4xl md:text-5xl font-bold"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              RackTrack
            </span>{" "}
            Visual Overview
          </motion.h2>

          <motion.p 
            className="max-w-4xl mx-auto text-center text-white/70 text-lg"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            Where AI Reveals Every Detail of Your Infrastructure
          </motion.p>
        </div>

        {/* Carousel Container */}
        <div className="relative">
          {/* Center highlight glow */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            <div className="w-80 h-full bg-gradient-to-r from-transparent via-primary/10 to-transparent" />
          </div>

          {/* Gradient fade edges */}
          <div className="absolute left-0 top-0 bottom-0 w-32 md:w-48 bg-gradient-to-r from-black to-transparent z-20 pointer-events-none" />
          <div className="absolute right-0 top-0 bottom-0 w-32 md:w-48 bg-gradient-to-l from-black to-transparent z-20 pointer-events-none" />

          {/* Scrolling track with CSS animation */}
          <div className="overflow-hidden py-8">
            <div 
              className="flex gap-6 animate-scroll"
              style={{
                width: 'max-content',
              }}
            >
              {scrollItems.map((item, index) => (
                <div
                  key={`${item.id}-${index}`}
                  className="group cursor-pointer flex-shrink-0 card-item"
                  data-testid={`gallery-item-${item.category}-${index}`}
                >
                  <div 
                    className="relative w-64 h-80 md:w-72 md:h-96 rounded-2xl overflow-hidden transition-all duration-500 group-hover:scale-105"
                    style={{
                      boxShadow: '0 20px 50px -15px rgba(0, 0, 0, 0.5)',
                    }}
                  >
                    {/* Image */}
                    <img 
                      src={item.image} 
                      alt={item.title}
                      className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                      data-testid={`gallery-image-${item.category}`}
                    />

                    {/* Gradient overlay */}
                    <div 
                      className="absolute inset-0 transition-opacity duration-500 opacity-60 group-hover:opacity-90"
                      style={{
                        background: `linear-gradient(to top, ${item.glowColor.replace('0.6', '0.98')} 0%, ${item.glowColor.replace('0.6', '0.5')} 35%, transparent 70%)`
                      }}
                    />

                    {/* Content overlay */}
                    <div className="absolute inset-0 flex flex-col justify-end p-5">
                      <h3 
                        className="text-white font-bold text-lg md:text-xl mb-2 transition-transform duration-400"
                        style={{ textShadow: '0 2px 10px rgba(0,0,0,0.8)' }}
                      >
                        {item.title}
                      </h3>

                      <p 
                        className="text-white/80 text-sm leading-relaxed opacity-0 translate-y-4 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-500"
                        style={{ textShadow: '0 1px 5px rgba(0,0,0,0.5)' }}
                      >
                        {item.description}
                      </p>
                    </div>

                    {/* Glowing border on hover */}
                    <div 
                      className="absolute inset-0 rounded-2xl pointer-events-none transition-opacity duration-500 opacity-0 group-hover:opacity-100"
                      style={{
                        boxShadow: `inset 0 0 0 2px ${item.glowColor}, 0 0 30px ${item.glowColor}`,
                      }}
                    />

                    {/* Top gradient accent on hover */}
                    <div 
                      className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${item.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}
                      style={{ boxShadow: `0 0 15px ${item.glowColor}` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CSS Animation */}
      <style>{`
        @keyframes scroll {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(calc(-100% / 3));
          }
        }

        .animate-scroll {
          animation: scroll 30s linear infinite;
        }

        .animate-scroll:hover {
          animation-play-state: running;
        }

        /* Center card emphasis using CSS */
        .card-item {
          transition: transform 0.4s ease, filter 0.4s ease;
        }
      `}</style>
    </section>
  );
};

export default Gallery;
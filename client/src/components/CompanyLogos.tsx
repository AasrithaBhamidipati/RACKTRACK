import { SiCisco, SiOracle, SiDell, SiHp, SiLenovo, SiVmware, SiIntel, SiAmd, SiNvidia, SiNetapp, SiRedhat } from "react-icons/si";
import { motion } from "framer-motion";
import darkGradientBg from "@assets/Screenshot 2025-09-25 162657_1759214694717.png";
import networkBg from "@assets/structural-connection-information-tech-coverage-generate-ai_98402-24492_1759214694717.jpg";

const CompanyLogos = () => {
  const companies = [
    { name: "cisco", Icon: SiCisco, displayName: "Cisco", color: "#049fd9" },
    { name: "oracle", Icon: SiOracle, displayName: "Oracle", color: "#c74634" },
    { name: "dell", Icon: SiDell, displayName: "Dell Technologies", color: "#007db8" },
    { name: "hp", Icon: SiHp, displayName: "HP", color: "#0096d6" },
    { name: "lenovo", Icon: SiLenovo, displayName: "Lenovo", color: "#E2231A" },
    { name: "vmware", Icon: SiVmware, displayName: "VMware", color: "#607078" },
    { name: "intel", Icon: SiIntel, displayName: "Intel", color: "#0071C5" },
    { name: "amd", Icon: SiAmd, displayName: "AMD", color: "#ED1C24" },
    { name: "nvidia", Icon: SiNvidia, displayName: "NVIDIA", color: "#76B900" },
    { name: "netapp", Icon: SiNetapp, displayName: "NetApp", color: "#0067C5" },
    { name: "redhat", Icon: SiRedhat, displayName: "Red Hat", color: "#EE0000" },
  ];

  // Duplicate the array to create seamless infinite scroll
  const duplicatedCompanies = [...companies, ...companies, ...companies];

  return (
    <section className="py-8 relative overflow-hidden">
      {/* Enhanced 3D Animated Background - Same as Hero but Darker */}
      <div className="absolute inset-0">
        {/* Base gradient background */}
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${darkGradientBg})` }}
        />
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url(${networkBg})` }}
        />
        {/* Darker overlay than hero */}
        <div className="absolute inset-0 bg-slate-950/75" />

        {/* 3D Animated Grid Layers - Subtler */}
        <motion.div
          className="absolute inset-0 opacity-15"
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

        {/* Secondary diagonal grid - Subtler */}
        <motion.div
          className="absolute inset-0 opacity-8"
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

        {/* Floating particles - Subtler than hero */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0.05, 0.3, 0.05],
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

        {/* Animated gradient orbs - Subtler */}
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-gradient-to-br from-cyan-500/15 to-transparent rounded-full blur-3xl"
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
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-gradient-to-br from-purple-500/15 to-transparent rounded-full blur-3xl"
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
      
      <div className="relative z-10 max-w-[1600px] mx-auto px-4 lg:px-6">
        <div className="relative overflow-hidden rounded-xl">
          {/* Fade edges */}
          <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-slate-950 to-transparent z-10"></div>
          <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-slate-950 to-transparent z-10"></div>
          
          <div className="flex animate-scroll-x">
            {duplicatedCompanies.map((company, index) => (
              <div 
                key={index}
                className="flex items-center justify-center min-w-fit px-6 md:px-8"
                data-testid={`logo-${company.name}-${index}`}
              >
                <motion.div 
                  className="flex items-center gap-3 px-6 py-3 rounded-xl backdrop-blur-xl border group hover-elevate transition-all duration-300"
                  style={{
                    background: `linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.12) 100%)`,
                    borderColor: company.color + "40",
                    boxShadow: `0 0 30px ${company.color}40, inset 0 0 20px ${company.color}15`
                  }}
                  whileHover={{ 
                    scale: 1.08,
                    boxShadow: `0 0 50px ${company.color}80, inset 0 0 30px ${company.color}40`,
                  }}
                  animate={{
                    boxShadow: [
                      `0 0 30px ${company.color}40, inset 0 0 20px ${company.color}15`,
                      `0 0 45px ${company.color}60, inset 0 0 30px ${company.color}25`,
                      `0 0 30px ${company.color}40, inset 0 0 20px ${company.color}15`,
                    ]
                  }}
                  transition={{
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <motion.div
                    animate={{
                      filter: [
                        `drop-shadow(0 0 8px ${company.color}60)`,
                        `drop-shadow(0 0 12px ${company.color}80)`,
                        `drop-shadow(0 0 8px ${company.color}60)`
                      ]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <company.Icon 
                      className="w-7 h-7 md:w-8 md:h-8 group-hover:scale-110 transition-transform" 
                      style={{ color: company.color }}
                    />
                  </motion.div>
                  <span 
                    className="text-sm font-bold whitespace-nowrap"
                    style={{ color: company.color, textShadow: `0 0 12px ${company.color}50` }}
                  >
                    {company.displayName}
                  </span>
                </motion.div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default CompanyLogos;

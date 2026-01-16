import { motion } from "framer-motion";
import { Cable, Cpu, Network, BarChart3, Server, Plug } from "lucide-react";

const FloatingReportsAnimation = () => {
  const floatingPanels = [
    {
      id: "cables",
      title: "CABLES REPORT",
      icon: Cable,
      position: { top: "5%", left: "-5%" },
      size: "w-40 h-28",
      delay: 0,
      content: (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-gradient-to-r from-orange-400 to-yellow-400 rounded-full" />
            <span className="text-[8px] text-cyan-300">CAT6</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-12 h-1.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full" />
            <span className="text-[8px] text-cyan-300">SFP+</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-10 h-1.5 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full" />
            <span className="text-[8px] text-cyan-300">QSFP</span>
          </div>
          <div className="grid grid-cols-3 gap-1 mt-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-1 bg-cyan-400/30 rounded" />
            ))}
          </div>
        </div>
      ),
    },
    {
      id: "switch",
      title: "SWITCH REPORT",
      icon: Network,
      position: { top: "0%", left: "35%" },
      size: "w-36 h-24",
      delay: 0.15,
      content: (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[8px] text-cyan-300">
            <span>PORT</span>
            <span>STATUS</span>
            <span>SPEED</span>
          </div>
          {["01", "02", "03"].map((port, i) => (
            <div key={i} className="flex justify-between items-center text-[7px]">
              <span className="text-white/80">{port}</span>
              <div className={`w-2 h-2 rounded-full ${i === 1 ? 'bg-yellow-400' : 'bg-green-400'}`} />
              <span className="text-cyan-300">10G</span>
            </div>
          ))}
        </div>
      ),
    },
    {
      id: "port",
      title: "PORT REPORT",
      icon: Plug,
      position: { top: "8%", right: "-8%" },
      size: "w-36 h-28",
      delay: 0.3,
      content: (
        <div className="space-y-1">
          <div className="grid grid-cols-4 gap-0.5">
            {[...Array(16)].map((_, i) => (
              <motion.div 
                key={i} 
                className="w-3 h-3 rounded-sm border border-cyan-400/40"
                style={{
                  background: i % 3 === 0 ? 'rgba(34, 211, 238, 0.3)' : 
                             i % 4 === 0 ? 'rgba(250, 204, 21, 0.3)' : 'transparent'
                }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, delay: i * 0.1, repeat: Infinity }}
              />
            ))}
          </div>
          <div className="flex justify-between text-[7px] text-cyan-300 mt-1">
            <span>Active: 12</span>
            <span>Idle: 4</span>
          </div>
        </div>
      ),
    },
    {
      id: "overall",
      title: "OVERALL REPORT",
      icon: BarChart3,
      position: { bottom: "15%", left: "-10%" },
      size: "w-44 h-32",
      delay: 0.45,
      content: (
        <div className="space-y-2">
          <div className="flex items-end gap-1 h-12">
            {[65, 80, 45, 90, 70, 55, 85].map((h, i) => (
              <motion.div
                key={i}
                className="flex-1 bg-gradient-to-t from-cyan-500 to-cyan-300 rounded-t-sm"
                style={{ height: `${h}%` }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: 0.5 + i * 0.1, duration: 0.5 }}
              />
            ))}
          </div>
          <div className="grid grid-cols-2 gap-2 text-[7px]">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-cyan-300">Detected: 127</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-yellow-400" />
              <span className="text-cyan-300">Warnings: 3</span>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "racktrack",
      title: "RACKTRACK",
      icon: Server,
      position: { bottom: "5%", right: "-5%" },
      size: "w-48 h-40",
      delay: 0.2,
      isMain: true,
      content: (
        <div className="space-y-2">
          <div className="relative w-full h-20 border border-cyan-400/40 rounded bg-slate-900/50 p-1.5">
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                className="w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
                animate={{ y: [-30, 30] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>
            <div className="grid grid-cols-4 gap-0.5 h-full">
              {[...Array(16)].map((_, i) => (
                <motion.div
                  key={i}
                  className="bg-slate-800 border border-cyan-400/20 rounded-sm flex items-center justify-center"
                  animate={{ 
                    borderColor: ['rgba(34, 211, 238, 0.2)', 'rgba(34, 211, 238, 0.6)', 'rgba(34, 211, 238, 0.2)']
                  }}
                  transition={{ duration: 1.5, delay: i * 0.05, repeat: Infinity }}
                >
                  <div className={`w-1 h-1 rounded-full ${i % 2 === 0 ? 'bg-cyan-400' : 'bg-green-400'}`} />
                </motion.div>
              ))}
            </div>
          </div>
          <div className="flex justify-between text-[8px]">
            <span className="text-cyan-300">Scan Progress</span>
            <span className="text-white">87%</span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-cyan-300 rounded-full"
              initial={{ width: "0%" }}
              animate={{ width: "87%" }}
              transition={{ duration: 2, delay: 0.5 }}
            />
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="relative w-full h-[380px] md:h-[450px]">
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          className="w-64 h-64 rounded-full bg-gradient-to-r from-cyan-500/20 via-cyan-400/10 to-cyan-500/20 blur-3xl"
          animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
      </div>

      {floatingPanels.map((panel) => {
        const Icon = panel.icon;
        return (
          <motion.div
            key={panel.id}
            className={`absolute ${panel.size}`}
            style={panel.position}
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ 
              opacity: 1, 
              scale: 1, 
              y: [0, -8, 0]
            }}
            transition={{
              opacity: { duration: 0.5, delay: panel.delay },
              scale: { duration: 0.5, delay: panel.delay },
              y: { 
                duration: 3 + panel.delay, 
                repeat: Infinity, 
                ease: "easeInOut",
                delay: panel.delay 
              }
            }}
            data-testid={`panel-${panel.id}`}
          >
            <div 
              className={`relative h-full rounded-lg overflow-hidden backdrop-blur-sm ${
                panel.isMain ? 'p-3' : 'p-2'
              }`}
              style={{
                background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%)',
                border: `1.5px solid ${panel.isMain ? 'rgba(34, 211, 238, 0.6)' : 'rgba(34, 211, 238, 0.4)'}`,
                boxShadow: `0 0 20px ${panel.isMain ? 'rgba(34, 211, 238, 0.3)' : 'rgba(34, 211, 238, 0.15)'},
                           inset 0 0 30px rgba(34, 211, 238, 0.05)`
              }}
            >
              <motion.div
                className="absolute inset-0 rounded-lg"
                style={{
                  background: 'linear-gradient(180deg, rgba(34, 211, 238, 0.1) 0%, transparent 50%)'
                }}
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{ duration: 2, repeat: Infinity }}
              />

              <div className="relative z-10">
                <div className="flex items-center gap-1.5 mb-2">
                  <Icon className="w-3 h-3 text-cyan-400" />
                  <span className="text-[9px] font-bold text-cyan-300 tracking-wider">
                    {panel.title}
                  </span>
                </div>
                {panel.content}
              </div>

              <div 
                className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan-400/60"
                style={{ borderTopLeftRadius: '8px' }}
              />
              <div 
                className="absolute top-0 right-0 w-2 h-2 border-t border-r border-cyan-400/60"
                style={{ borderTopRightRadius: '8px' }}
              />
              <div 
                className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-cyan-400/60"
                style={{ borderBottomLeftRadius: '8px' }}
              />
              <div 
                className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan-400/60"
                style={{ borderBottomRightRadius: '8px' }}
              />
            </div>
          </motion.div>
        );
      })}

      <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
        <motion.line
          x1="20%" y1="30%" x2="45%" y2="50%"
          stroke="rgba(34, 211, 238, 0.3)"
          strokeWidth="1"
          strokeDasharray="4 4"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        />
        <motion.line
          x1="50%" y1="15%" x2="55%" y2="45%"
          stroke="rgba(34, 211, 238, 0.3)"
          strokeWidth="1"
          strokeDasharray="4 4"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 0.7 }}
        />
        <motion.line
          x1="80%" y1="25%" x2="70%" y2="50%"
          stroke="rgba(34, 211, 238, 0.3)"
          strokeWidth="1"
          strokeDasharray="4 4"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 0.9 }}
        />
        <motion.line
          x1="15%" y1="75%" x2="40%" y2="65%"
          stroke="rgba(34, 211, 238, 0.3)"
          strokeWidth="1"
          strokeDasharray="4 4"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 1.1 }}
        />
      </svg>
    </div>
  );
};

export default FloatingReportsAnimation;

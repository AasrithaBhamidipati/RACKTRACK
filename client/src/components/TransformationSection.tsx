import { useState, useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { 
  Clock, 
  Zap, 
  FileSpreadsheet, 
  Camera, 
  Brain, 
  FileCheck,
  ArrowRight,
  Check,
  X,
  Timer,
  Sparkles
} from "lucide-react";

const TransformationSection = () => {
  const [activeComparison, setActiveComparison] = useState<'before' | 'after'>('before');
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const morphProgress = useTransform(scrollYProgress, [0.2, 0.5], [0, 1]);

  const beforeSteps = [
    { icon: Clock, label: "12+ Hours", desc: "Per rack audit", color: "#ef4444" },
    { icon: FileSpreadsheet, label: "Manual Entry", desc: "Spreadsheet updates", color: "#f97316" },
    { icon: X, label: "Human Errors", desc: "Inconsistent data", color: "#eab308" },
  ];

  const afterSteps = [
    { icon: Camera, label: "Scan", desc: "Point & capture", color: "#22d3ee" },
    { icon: Brain, label: "AI Process", desc: "2-5 seconds", color: "#a855f7" },
    { icon: FileCheck, label: "Report", desc: "Auto-generated", color: "#22c55e" },
  ];

  return (
    <section 
      ref={containerRef}
      className="py-24 relative overflow-hidden bg-gradient-to-b from-black via-slate-950 to-black"
    >
      <div className="absolute inset-0">
        <motion.div 
          className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full blur-3xl"
          style={{
            background: activeComparison === 'before' 
              ? 'rgba(239, 68, 68, 0.15)' 
              : 'rgba(34, 211, 238, 0.15)'
          }}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full blur-3xl"
          style={{
            background: activeComparison === 'before'
              ? 'rgba(249, 115, 22, 0.1)'
              : 'rgba(168, 85, 247, 0.1)'
          }}
          animate={{ scale: [1.2, 1, 1.2] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
      </div>

      <div className="max-w-[1600px] mx-auto px-4 lg:px-6 relative z-10">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div className="inline-flex items-center px-4 py-2 rounded-full border border-cyan-400/40 bg-cyan-500/10 mb-6">
            <Sparkles className="w-4 h-4 text-cyan-400 mr-2" />
            <span className="text-cyan-300 text-sm font-medium">THE TRANSFORMATION</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-bold mb-4">
            <span className="text-white">From </span>
            <motion.span 
              className="bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent"
              animate={{ opacity: activeComparison === 'before' ? 1 : 0.5 }}
            >
              Hours
            </motion.span>
            <span className="text-white"> to </span>
            <motion.span 
              className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
              animate={{ opacity: activeComparison === 'after' ? 1 : 0.5 }}
            >
              Seconds
            </motion.span>
          </h2>
          <p className="text-white/60 text-lg max-w-2xl mx-auto">
            See how RackTrack revolutionizes infrastructure auditing
          </p>
        </motion.div>

        <div className="flex justify-center mb-12">
          <div className="inline-flex rounded-full p-1 bg-slate-900/80 border border-white/10">
            <button
              onClick={() => setActiveComparison('before')}
              className={`px-8 py-3 rounded-full text-sm font-semibold transition-all duration-300 ${
                activeComparison === 'before'
                  ? 'bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-lg shadow-red-500/30'
                  : 'text-white/60 hover:text-white'
              }`}
              data-testid="button-before"
            >
              Manual Process
            </button>
            <button
              onClick={() => setActiveComparison('after')}
              className={`px-8 py-3 rounded-full text-sm font-semibold transition-all duration-300 ${
                activeComparison === 'after'
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg shadow-cyan-500/30'
                  : 'text-white/60 hover:text-white'
              }`}
              data-testid="button-after"
            >
              With RackTrack
            </button>
          </div>
        </div>

        <div className="relative">
          <motion.div
            className="absolute inset-0 flex items-center justify-center pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: activeComparison === 'after' ? 1 : 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              className="w-32 h-32 rounded-full bg-gradient-to-r from-cyan-500/30 to-purple-500/30 blur-2xl"
              animate={{ scale: [1, 1.5, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </motion.div>

          <motion.div 
            className="grid md:grid-cols-3 gap-6"
            key={activeComparison}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {(activeComparison === 'before' ? beforeSteps : afterSteps).map((step, index) => {
              const Icon = step.icon;
              return (
                <motion.div
                  key={step.label}
                  className="relative group"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.15 }}
                  data-testid={`card-step-${index}`}
                >
                  <div 
                    className="relative rounded-2xl p-6 h-full backdrop-blur-sm overflow-hidden"
                    style={{
                      background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%)',
                      border: `1px solid ${step.color}40`,
                    }}
                  >
                    <motion.div
                      className="absolute -top-20 -right-20 w-40 h-40 rounded-full opacity-20 blur-2xl"
                      style={{ background: step.color }}
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 3, repeat: Infinity, delay: index * 0.3 }}
                    />

                    <div className="relative z-10">
                      <motion.div
                        className="w-16 h-16 rounded-xl flex items-center justify-center mb-4"
                        style={{
                          background: `${step.color}20`,
                          border: `1px solid ${step.color}40`,
                        }}
                        whileHover={{ scale: 1.1, rotate: 5 }}
                      >
                        <Icon className="w-8 h-8" style={{ color: step.color }} />
                      </motion.div>

                      <h3 className="text-2xl font-bold text-white mb-2">{step.label}</h3>
                      <p className="text-white/60">{step.desc}</p>

                      <motion.div
                        className="absolute bottom-0 left-0 right-0 h-1 rounded-full"
                        style={{ background: step.color }}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: 1 }}
                        transition={{ delay: index * 0.2 + 0.3, duration: 0.5 }}
                      />
                    </div>
                  </div>

                  {index < 2 && (
                    <div className="hidden md:flex absolute -right-3 top-1/2 -translate-y-1/2 z-20">
                      <motion.div
                        className="w-6 h-6 rounded-full flex items-center justify-center"
                        style={{
                          background: activeComparison === 'before' 
                            ? 'linear-gradient(135deg, #ef4444, #f97316)' 
                            : 'linear-gradient(135deg, #22d3ee, #a855f7)',
                        }}
                        animate={{ x: [0, 5, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        <ArrowRight className="w-3 h-3 text-white" />
                      </motion.div>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </motion.div>
        </div>

        <motion.div
          className="mt-16 grid md:grid-cols-2 gap-8"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
        >
          <div 
            className={`relative rounded-2xl p-6 transition-all duration-500 ${
              activeComparison === 'before' 
                ? 'ring-2 ring-red-500/50' 
                : 'opacity-60'
            }`}
            style={{
              background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
            }}
          >
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                <Timer className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h4 className="text-xl font-bold text-white">Old Way</h4>
                <p className="text-red-400 text-sm">Time-consuming & error-prone</p>
              </div>
            </div>
            <div className="space-y-3">
              {["12+ hours per rack", "Manual spreadsheet updates", "Prone to human errors", "Delayed documentation", "Inconsistent formatting"].map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <X className="w-4 h-4 text-red-400 flex-shrink-0" />
                  <span className="text-white/70 text-sm">{item}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-red-500/20">
              <div className="text-3xl font-bold text-red-400">360+ hours</div>
              <div className="text-white/50 text-sm">For 30-rack facility</div>
            </div>
          </div>

          <div 
            className={`relative rounded-2xl p-6 transition-all duration-500 ${
              activeComparison === 'after' 
                ? 'ring-2 ring-cyan-500/50' 
                : 'opacity-60'
            }`}
            style={{
              background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%)',
              border: '1px solid rgba(34, 211, 238, 0.3)',
            }}
          >
            <div className="flex items-center gap-4 mb-4">
              <motion.div 
                className="w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center"
                animate={{ boxShadow: ['0 0 0 0 rgba(34, 211, 238, 0)', '0 0 0 8px rgba(34, 211, 238, 0.1)', '0 0 0 0 rgba(34, 211, 238, 0)'] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Zap className="w-6 h-6 text-cyan-400" />
              </motion.div>
              <div>
                <h4 className="text-xl font-bold text-white">With RackTrack</h4>
                <p className="text-cyan-400 text-sm">Fast, accurate & automated</p>
              </div>
            </div>
            <div className="space-y-3">
              {["2-5 seconds per rack scan", "Auto-generated reports", "95%+ detection accuracy", "Instant documentation", "Professional formatting"].map((item, i) => (
                <motion.div 
                  key={i} 
                  className="flex items-center gap-3"
                  initial={{ x: -10, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <Check className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                  <span className="text-white/70 text-sm">{item}</span>
                </motion.div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-cyan-500/20">
              <div className="flex items-baseline gap-2">
                <div className="text-3xl font-bold text-cyan-400">15 hours</div>
                <div className="text-green-400 text-sm font-medium">-96%</div>
              </div>
              <div className="text-white/50 text-sm">For 30-rack facility</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default TransformationSection;

import { useEffect, useState, useRef } from "react";
import { useInView } from "framer-motion";
import { motion } from "framer-motion";
import { Server, Target, Clock, Users } from "lucide-react";

const StatsSection = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const [counts, setCounts] = useState({
    racks: 0,
    accuracy: 0,
    timeSaved: 0,
    users: 0,
  });

  const stats = [
    {
      icon: Server,
      label: "Racks Analyzed",
      value: 10000,
      suffix: "+",
      color: "from-blue-400 to-cyan-400",
    },
    {
      icon: Target,
      label: "Detection Accuracy",
      value: 99.8,
      suffix: "%",
      color: "from-emerald-400 to-green-400",
    },
    {
      icon: Clock,
      label: "Hours Saved",
      value: 50000,
      suffix: "+",
      color: "from-violet-400 to-purple-400",
    },
    {
      icon: Users,
      label: "Active Users",
      value: 5000,
      suffix: "+",
      color: "from-pink-400 to-rose-400",
    },
  ];

  useEffect(() => {
    if (!isInView) return;

    const duration = 2000;
    const steps = 60;
    const interval = duration / steps;

    let currentStep = 0;

    const timer = setInterval(() => {
      currentStep++;
      const progress = currentStep / steps;

      setCounts({
        racks: Math.floor(stats[0].value * progress),
        accuracy: Number((stats[1].value * progress).toFixed(1)),
        timeSaved: Math.floor(stats[2].value * progress),
        users: Math.floor(stats[3].value * progress),
      });

      if (currentStep >= steps) {
        clearInterval(timer);
      }
    }, interval);

    return () => clearInterval(timer);
  }, [isInView]);

  return (
    <section className="relative py-24 overflow-hidden">
      {/* Glass-morphism background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900/50 via-slate-800/50 to-slate-900/50 backdrop-blur-3xl"></div>

      <div className="relative z-10 max-w-[1600px] mx-auto px-4 lg:px-6" ref={ref}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4 text-center">
            Trusted by{" "}
            <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent animate-gradient">
              Industry Leaders
            </span>
          </h2>
          <p className="text-white/70 text-lg">
            Delivering results that speak for themselves
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="group relative"
              data-testid={`stat-${stat.label.toLowerCase().replace(/\s+/g, "-")}`}
            >
              {/* Glass card with gradient border */}
              <div className="relative bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:border-white/20">
                {/* Gradient glow effect */}
                <div className={`absolute -inset-0.5 bg-gradient-to-r ${stat.color} rounded-2xl opacity-0 group-hover:opacity-20 blur transition-opacity duration-300`}></div>

                <div className="relative">
                  {/* Icon */}
                  <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br ${stat.color} mb-4`}>
                    <stat.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Value */}
                  <div className="mb-2">
                    <span className={`text-4xl md:text-5xl font-bold bg-gradient-to-r ${stat.color} bg-clip-text text-transparent`}>
                      {index === 0 ? counts.racks.toLocaleString() :
                       index === 1 ? counts.accuracy :
                       index === 2 ? counts.timeSaved.toLocaleString() :
                       counts.users.toLocaleString()}
                      {stat.suffix}
                    </span>
                  </div>

                  {/* Label */}
                  <p className="text-white/70 text-sm font-medium">
                    {stat.label}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
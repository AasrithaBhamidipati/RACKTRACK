import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";
import robotImage from "@assets/Screenshot_2025-09-25_153928-removebg-preview_1759221704615.png";
import backgroundImage from "@assets/Screenshot 2025-09-25 162657_1759152093398.png";
import { ArrowRight } from "lucide-react";

const JoinToday = () => {
  const [, setLocation] = useLocation();

  return (
    <section className="relative overflow-hidden py-24">
      {/* Background image */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${backgroundImage})` }}
        />
        <div className="absolute inset-0 bg-slate-950/60" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-6">
            {/* JOIN TODAY Badge */}
            <div className="inline-block">
              <div
                className="inline-flex items-center px-6 py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg shadow-2xl shadow-primary/30 relative overflow-hidden group"
                data-testid="text-join-today"
              >
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/0 via-cyan-400/30 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative text-white font-bold tracking-widest uppercase" style={{ fontSize: "14px", letterSpacing: "0.1em" }}>
                  Join Today
                </span>
              </div>
            </div>

            {/* Main Heading */}
            <h2 className="text-white text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-center lg:text-left">
              <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Ready to Transform
              </span>{" "}
              Your Infrastructure Management?
            </h2>

            {/* Description */}
            <p className="max-w-lg text-lg text-slate-300">
              Join IT teams saving 40+ hours per month on rack auditing. Get started with a free trial - no credit card required
            </p>

            {/* CTA Button */}
            <div className="pt-4">
              <Button
                size="default"
                variant="default"
                className="group bg-gradient-to-r from-primary to-cyan-500 hover:from-primary/90 hover:to-cyan-600 text-white border-0 shadow-2xl hover:shadow-primary/50 transition-all duration-300"
                data-testid="button-start-free-trial"
                onClick={() => setLocation("/coming-soon")}
              >
                Start Free Trial →
              </Button>
            </div>
          </div>

          {/* Right Robot Image */}
          <div className="relative flex justify-center lg:justify-end">
            <div className="relative">
              {/* Robot Image from uploaded assets */}
              <img
                src={robotImage}
                alt="AI Robot Assistant with glowing blue eyes"
                className="relative z-10 w-[350px] h-auto md:w-[450px] lg:w-[500px] xl:w-[550px] object-contain"
                data-testid="img-robot-assistant"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default JoinToday;
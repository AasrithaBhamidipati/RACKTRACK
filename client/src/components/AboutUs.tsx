import { CheckCircle } from "lucide-react";
import teamImage from "@assets/image_1764767868070.png";
import aboutUsBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";

const AboutUs = () => {
  return (
    <section id="about-us" className="relative overflow-hidden py-24">
      {/* Background */}
      <div className="absolute inset-0">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${aboutUsBg})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main About Section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">

          {/* Left side - Large Image with glow effect */}
          <div className="lg:col-span-7 flex justify-center">
            <div className="relative">
              {/* Glow effect behind image */}
              <div className="absolute -inset-8 bg-gradient-to-r from-cyan-500/20 via-primary/30 to-purple-500/20 rounded-full blur-3xl"></div>
              <img 
                src={teamImage}
                alt="Data center professionals working together"
                className="relative w-full max-w-[600px] h-[450px] md:h-[550px] object-cover rounded-lg drop-shadow-2xl"
                data-testid="img-about-us"
              />
            </div>
          </div>

          {/* Right side - Content */}
          <div className="lg:col-span-5 text-left">
            {/* About Us Badge */}
            <div className="mb-6 flex justify-start">
              <div className="inline-flex items-center px-6 py-3 rounded-full border border-primary/60 bg-gradient-to-r from-primary/20 via-cyan-500/15 to-primary/20 backdrop-blur-lg shadow-2xl shadow-primary/30 relative overflow-hidden group">
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/0 via-cyan-400/30 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative text-white font-bold tracking-widest uppercase" style={{ fontSize: "14px", letterSpacing: "0.1em" }}>
                  About Us
                </span>
              </div>
            </div>

            <h2 className="text-white mb-6 text-left font-semibold tracking-wide" style={{ fontFamily: "'Inter', 'Poppins', sans-serif" }}>
              <span className="bg-gradient-to-r from-primary via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Smart Infrastructure Auditing
              </span>{" "}
              for Modern Data Centers
            </h2>

            <p className="mb-6 text-left" style={{ fontSize: "18px" }}>
              Traditional manual rack audits consume 12+ hours per rack. RackTrack automates the entire process using computer vision and AI - completing what used to take weeks in just Minutes
            </p>

            <div className="space-y-4 mb-8">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-5 h-5 text-primary" strokeWidth={2} />
                </div>
                <p style={{ fontSize: "16px" }}>
                  <strong className="text-white">Manual Process:</strong> ~12 hours per rack - identifying devices, tracing cables, reading labels, updating spreadsheets. For a 30-rack facility, that's 360 hours of work
                </p>
              </div>

              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-5 h-5 text-cyan-400" strokeWidth={2} />
                </div>
                <p style={{ fontSize: "16px" }}>
                  <strong className="text-white">With RackTrack:</strong> Scan racks with mobile/tablet, automatically identify all components, read labels via OCR, map cable connections, validate network topology
                </p>
              </div>

              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  <CheckCircle className="w-5 h-5 text-purple-400" strokeWidth={2} />
                </div>
                <p style={{ fontSize: "16px" }}>
                  <strong className="text-white">Result:</strong> 360 hours → 15 hours (95% reduction). Complete documentation, change tracking, professional audit reports
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutUs;
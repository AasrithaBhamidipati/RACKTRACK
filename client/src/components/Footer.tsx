import { Link } from "wouter";
import {
  Instagram,
  Linkedin,
  Facebook,
  Mail,
  Phone,
  MapPin,
} from "lucide-react";
import darkGradientBg from "@assets/Screenshot 2025-09-25 162657_1759149191478.png";
import networkBg from "@assets/structural-connection-information-tech-coverage-generate-ai_98402-24492_1759214694717.jpg";
import logoImage from "@assets/image_1761831098460.png";

const Footer = () => {
  const navItems = [
    { path: "/", label: "Home" },
    { path: "/solutions", label: "Solutions" },
    { path: "/product", label: "Features" },
    { path: "/about", label: "About" },
  ];

  return (
    <footer className="relative overflow-hidden">
      {/* Same background as hero section */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${darkGradientBg})` }}
        />
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url(${networkBg})` }}
        />
        <div className="absolute inset-0 bg-slate-950/60" />
      </div>

      {/* Creative top design - flowing particles and gradient beams */}
      <div className="absolute top-0 left-0 right-0 h-32 overflow-hidden">
        {/* Gradient beams */}
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-0 left-0 w-1/3 h-24 bg-gradient-to-br from-cyan-500/20 via-transparent to-transparent blur-2xl"></div>
          <div className="absolute top-0 right-0 w-1/3 h-24 bg-gradient-to-bl from-blue-500/20 via-transparent to-transparent blur-2xl"></div>
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-20 bg-gradient-to-b from-purple-500/10 via-transparent to-transparent blur-xl"></div>
        </div>

        {/* Flowing wave path */}
        <svg
          className="absolute top-0 left-0 w-full h-full"
          viewBox="0 0 1440 120"
          preserveAspectRatio="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient
              id="footerGradient1"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="rgba(6, 182, 212, 0.4)" />
              <stop offset="50%" stopColor="rgba(59, 130, 246, 0.6)" />
              <stop offset="100%" stopColor="rgba(168, 85, 247, 0.4)" />
            </linearGradient>
            <linearGradient
              id="footerGradient2"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="rgba(6, 182, 212, 0.2)" />
              <stop offset="50%" stopColor="rgba(59, 130, 246, 0.3)" />
              <stop offset="100%" stopColor="rgba(168, 85, 247, 0.2)" />
            </linearGradient>
          </defs>
          {/* Multiple flowing waves */}
          <path
            d="M0,60 Q360,30 720,60 T1440,60 L1440,0 L0,0 Z"
            fill="url(#footerGradient1)"
            opacity="0.6"
          >
            <animate
              attributeName="d"
              dur="8s"
              repeatCount="indefinite"
              values="
                M0,60 Q360,30 720,60 T1440,60 L1440,0 L0,0 Z;
                M0,45 Q360,75 720,45 T1440,45 L1440,0 L0,0 Z;
                M0,60 Q360,30 720,60 T1440,60 L1440,0 L0,0 Z
              "
            />
          </path>
          <path
            d="M0,40 Q360,70 720,40 T1440,40 L1440,0 L0,0 Z"
            fill="url(#footerGradient2)"
            opacity="0.4"
          >
            <animate
              attributeName="d"
              dur="10s"
              repeatCount="indefinite"
              values="
                M0,40 Q360,70 720,40 T1440,40 L1440,0 L0,0 Z;
                M0,70 Q360,40 720,70 T1440,70 L1440,0 L0,0 Z;
                M0,40 Q360,70 720,40 T1440,40 L1440,0 L0,0 Z
              "
            />
          </path>
        </svg>

        {/* Animated particles/dots */}
        <div className="absolute top-10 left-[15%] w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(6,182,212,0.8)]"></div>
        <div
          className="absolute top-6 left-[35%] w-2 h-2 bg-blue-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(59,130,246,0.8)]"
          style={{ animationDelay: "0.5s" }}
        ></div>
        <div
          className="absolute top-14 left-[55%] w-1 h-1 bg-purple-400 rounded-full animate-pulse shadow-[0_0_6px_rgba(168,85,247,0.8)]"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="absolute top-8 right-[25%] w-2 h-2 bg-cyan-300 rounded-full animate-pulse shadow-[0_0_10px_rgba(103,232,249,0.8)]"
          style={{ animationDelay: "0.3s" }}
        ></div>
        <div
          className="absolute top-12 right-[45%] w-1.5 h-1.5 bg-blue-300 rounded-full animate-pulse shadow-[0_0_8px_rgba(147,197,253,0.8)]"
          style={{ animationDelay: "0.8s" }}
        ></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
        {/* Brand Identity - Top Left */}
            <div className="flex justify-start mb-12">
          <Link href="/">
            <div className="flex items-center gap-3 cursor-pointer group">
              <img
                src={logoImage}
                alt="RackTrack Logo"
                className="w-12 h-12 group-hover:brightness-110 transition-all"
                style={{
                  filter:
                    "invert(1) brightness(1.2) sepia(1) saturate(5) hue-rotate(180deg)",
                }}
                data-testid="footer-logo"
              />
              <div className="text-left">
                <h2
                  className="text-2xl font-normal tracking-wide uppercase transition-all duration-500 group-hover:tracking-wider"
                >
                  <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent drop-shadow-[0_2px_8px_rgba(6,182,212,0.5)]">
                    RACKTRACK
                  </span>
                </h2>
                <p className="text-white/60 text-xs mt-1">
                  AI-Powered Infrastructure Management for Modern Networks
                </p>
              </div>
            </div>
          </Link>
        </div>

        {/* 4 Columns Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-10 mb-12">
          {/* Column 1: Pages */}
          <div>
            <h3 className="text-white font-semibold text-sm uppercase tracking-wider mb-4">
              Pages
            </h3>
            <ul className="space-y-3">
              {navItems.map((item) => (
                <li key={item.path}>
                  <Link
                    href={item.path}
                    className="text-white/70 hover:text-cyan-400 transition-colors text-sm block"
                    data-testid={`footer-link-${item.label.toLowerCase()}`}
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Column 2: Contact */}
          <div>
            <h3
              className="text-white font-semibold text-sm uppercase tracking-wider mb-4"
              data-testid="footer-section-contact"
            >
              Contact
            </h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="mailto:info@sprintpark.com"
                  className="flex items-center gap-2 text-white/70 hover:text-cyan-400 transition-colors text-sm group"
                  data-testid="footer-email-info"
                >
                  <Mail className="w-4 h-4 text-cyan-400/70 group-hover:text-cyan-400" />
                  info@sprintpark.com
                </a>
              </li>
              <li>
                <a
                  href="mailto:spracktrack.ai@gmail.com"
                  className="flex items-center gap-2 text-white/70 hover:text-cyan-400 transition-colors text-sm group"
                  data-testid="footer-email-hr"
                >
                  <Mail className="w-4 h-4 text-cyan-400/70 group-hover:text-cyan-400" />
                  spracktrack.ai@gmail.com
                </a>
              </li>
              <li>
                <a
                  href="tel:+18605669894"
                  className="flex items-center gap-2 text-white/70 hover:text-cyan-400 transition-colors text-sm group mt-4"
                  data-testid="footer-phone-us"
                >
                  <Phone className="w-4 h-4 text-cyan-400/70 group-hover:text-cyan-400" />
                  +1 (860) 566 9894
                </a>
              </li>
              <li>
                <a
                  href="tel:+917207735554"
                  className="flex items-center gap-2 text-white/70 hover:text-cyan-400 transition-colors text-sm group"
                  data-testid="footer-phone-india"
                >
                  <Phone className="w-4 h-4 text-cyan-400/70 group-hover:text-cyan-400" />
                  +91 7207735554
                </a>
              </li>
            </ul>
          </div>

          {/* Column 3: Locations */}
          <div>
            <h3
              className="text-white font-semibold text-sm uppercase tracking-wider mb-4"
              data-testid="footer-section-location"
            >
              Locations
            </h3>
            <div className="space-y-6">
              <div>
                <p className="flex items-center gap-2 text-white/90 font-medium text-sm mb-2">
                  <MapPin className="w-4 h-4 text-cyan-400" />
                  United States
                </p>
                <p
                  className="text-white/60 text-xs leading-relaxed pl-6"
                  data-testid="footer-address-usa"
                >
                  85 Felt Rd, Suite #604
                  <br />
                  South Windsor, CT 06074
                </p>
              </div>
              <div>
                <p className="flex items-center gap-2 text-white/90 font-medium text-sm mb-2">
                  <MapPin className="w-4 h-4 text-blue-400" />
                  India
                </p>
                <p
                  className="text-white/60 text-xs leading-relaxed pl-6"
                  data-testid="footer-address-india"
                >
                  Asian Sun City, Unit 1204, Block B<br />
                  Kondapur, Hyderabad 500084
                </p>
              </div>
            </div>
          </div>

          {/* Column 4: Follow Us */}
          <div>
            <h3
              className="text-white font-semibold text-sm uppercase tracking-wider mb-4"
              data-testid="footer-section-social"
            >
              Follow Us
            </h3>
            <div className="space-y-4">
              <a
                href="https://www.linkedin.com/company/sprintpark-tech"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 text-white/70 hover:text-cyan-400 transition-all group"
                data-testid="footer-social-linkedin"
              >
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center group-hover:bg-cyan-500/10 transition-all">
                  <Linkedin className="w-5 h-5" />
                </div>
                <span className="text-sm">LinkedIn</span>
              </a>
              <a
                href="https://www.instagram.com/sprintpark/#"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 text-white/70 hover:text-cyan-400 transition-all group"
                data-testid="footer-social-instagram"
              >
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center group-hover:bg-cyan-500/10 transition-all">
                  <Instagram className="w-5 h-5" />
                </div>
                <span className="text-sm">Instagram</span>
              </a>
              <a
                href="https://www.facebook.com/SPRINTPARKLLC"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 text-white/70 hover:text-cyan-400 transition-all group"
                data-testid="footer-social-facebook"
              >
                <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center group-hover:bg-cyan-500/10 transition-all">
                  <Facebook className="w-5 h-5" />
                </div>
                <span className="text-sm">Facebook</span>
              </a>
            </div>
          </div>
        </div>

        {/* Divider with gradient */}
        <div className="relative h-px mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent"></div>
        </div>

        {/* Bottom Bar */}
        <div className="pb-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-white/40 text-sm">
            © {new Date().getFullYear()} SprintPark LLC. All rights reserved.
          </p>
          <div className="flex gap-6 text-sm">
            <a
              href="#"
              className="text-white/50 hover:text-white/70 transition-colors"
            >
              Privacy Policy
            </a>
            <a
              href="#"
              className="text-white/50 hover:text-white/70 transition-colors"
            >
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
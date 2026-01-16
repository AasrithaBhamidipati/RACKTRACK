import { Link, useLocation } from "wouter";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import logoImage from "@assets/image_1761831098460.png";

const baseTheme = {
  bg: "bg-black/95",
  border: "border-[#003399]/50",
  text: "text-white",
  activeText: "text-white",
  activeBg: "bg-[#003399]/20",
};
const theme = baseTheme;

const Navigation = () => {
  const [location, setLocation] = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [username, setUsername] = useState<string | null>(null);
  const [currentAvatar, setCurrentAvatar] = useState<string | null>(null);

  // minimal nav: no marketing links; navItems left empty so only auth actions render
  const navItems: { path: string; label: string }[] = []; 

  useEffect(() => {
    const checkAuth = () => setIsAuthenticated(localStorage.getItem("isAuthenticated") === "true");
    const updateAvatar = () => setCurrentAvatar(localStorage.getItem("profileImage"));
    const updateUsername = () => setUsername(localStorage.getItem('username'));
    checkAuth();
    updateAvatar();
    updateUsername();

    // verify server-side session if client thinks we're authenticated
    const verifyAuth = async () => {
      if (localStorage.getItem("isAuthenticated") !== "true") return;
      try {
        const res = await fetch("/api/user/profile", { credentials: "include" });
        if (!res.ok) {
          // server doesn't recognize the session -> clear stale client auth
          localStorage.removeItem("isAuthenticated");
          localStorage.removeItem("username");
          localStorage.removeItem("profileImage");
          setIsAuthenticated(false);
          setUsername(null);
          setCurrentAvatar(null);
          // notify other parts of the app
          window.dispatchEvent(new Event("auth-changed"));
        } else {
          // optionally update username from server
          try {
            const data = await res.json();
            const serverUsername = data?.user?.username || data?.username;
            if (serverUsername) {
              localStorage.setItem("username", serverUsername);
              setUsername(serverUsername);
            }
          } catch (e) {
            // ignore JSON parse errors
          }
          setIsAuthenticated(true);
        }
      } catch (e) {
        // network error — keep existing client state but don't assert authenticated
        // (could opt to setIsAuthenticated(false) here if desired)
        console.warn("verifyAuth failed:", e);
      }
    };

    verifyAuth();
    window.addEventListener("storage", checkAuth);
    window.addEventListener("storage", updateAvatar);
    window.addEventListener("storage", updateUsername);
    window.addEventListener("auth-changed", checkAuth as EventListener);
    window.addEventListener("profile-updated", updateAvatar as EventListener);
    window.addEventListener("profile-updated", updateUsername as EventListener);
    return () => {
      window.removeEventListener("storage", checkAuth);
      window.removeEventListener("storage", updateAvatar);
      window.removeEventListener("storage", updateUsername);
      window.removeEventListener("auth-changed", checkAuth as EventListener);
      window.removeEventListener("profile-updated", updateAvatar as EventListener);
      window.removeEventListener("profile-updated", updateUsername as EventListener);
    };
  }, []);


  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-[100] ${theme.bg} backdrop-blur-md border-b ${theme.border} shadow-lg transition-colors duration-300`}
    >
      <div className="max-w-[1600px] mx-auto px-4 lg:px-6 py-4">
        <div className="flex items-center justify-between">
          {/* ... (Your Logo/Brand JSX block) ... */}
          <Link href="/">
            <div
              className="flex items-center space-x-3 cursor-pointer group"
              data-testid="logo"
            >
              <img
                src={logoImage}
                alt="RackTrack Logo"
                className="w-12 h-12 group-hover:brightness-110 transition-all rounded-lg"
                style={{
                  filter: "invert(1) brightness(1.2) sepia(1) saturate(5) hue-rotate(180deg) drop-shadow(0 0 8px rgba(6,182,212,0.6))",
                  backgroundColor: "rgba(0, 0, 0, 0.6)",
                }}
                data-testid="icon-logo"
              />
              <span
                className="relative text-lg md:text-xl lg:text-2xl font-normal tracking-wide uppercase transition-all duration-500 group-hover:tracking-wider"
                data-testid="text-brand"
              >
                <span className="relative z-10 bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent drop-shadow-[0_2px_8px_rgba(6,182,212,0.5)]">
                  RACKTRACK
                </span>
              </span>
            </div>
          </Link>
          {/* ... (Your Desktop Nav Items JSX block) ... */}
          <div className="hidden lg:flex items-center space-x-1">
            <div className="flex items-center gap-1">
              {navItems.map((item) => {
                const isActive = location === item.path;

                return (
                  <Link key={item.path} href={item.path}>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`relative px-4 py-2.5 rounded-md text-sm font-semibold tracking-wide transition-all uppercase cursor-pointer flex items-center gap-1.5 group text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[#0066DD]`}
                      data-testid={`link-${item.label.toLowerCase().replace(' ', '-')}`}
                    >
                      <span className="relative z-10 font-semibold tracking-wide">
                        {isActive ? (
                          <span className="text-white drop-shadow-[0_0_12px_rgba(0,102,221,0.6)]">
                            {item.label}
                          </span>
                        ) : (
                          item.label
                        )}
                      </span>
                      <motion.div
                        className="absolute bottom-0 left-4 right-4 h-0.5 bg-gradient-to-r from-[#0066DD] to-[#003399] rounded-full"
                        style={{
                          boxShadow: isActive ? "0 0 12px rgba(0, 102, 221, 1), 0 0 24px rgba(0, 51, 153, 0.8)" : "none"
                        }}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: isActive ? 1 : 0 }}
                        whileHover={!isActive ? { scaleX: 1 } : {}}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.div>
                  </Link>
                );
              })}
            </div>

            {/* Auth actions: show sign up/sign in for guests, account avatar for authenticated users */}
            <div className="flex items-center gap-2 ml-4">
              {!isAuthenticated ? (
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => setLocation('/login')}
                    className="px-3 py-2 rounded-md text-sm font-semibold text-white hover:bg-white/5"
                    data-testid="button-signup-signin"
                  >
                    Sign up / Sign in
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2 ml-4">
                  <Link href="/account">
                    <div
                      className="w-9 h-9 rounded-full bg-white/10 flex items-center justify-center text-white font-semibold overflow-hidden"
                      aria-label="Account"
                      data-testid="nav-avatar-button"
                    >
                      {currentAvatar ? (
                        <img src={currentAvatar} alt="avatar" className="w-full h-full object-cover" />
                      ) : username ? (
                        <span>{username.slice(0, 1).toUpperCase()}</span>
                      ) : (
                        <span>U</span>
                      )}
                    </div>
                  </Link>
                </div>
              )}
            </div>
          </div>
          {/* ... (Your Mobile Menu Button JSX block) ... */}
          <div className="flex lg:hidden items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-white focus:outline-none"
              data-testid="mobile-menu-button"
              aria-controls="mobile-menu"
              aria-expanded={isMenuOpen}
              aria-label="Toggle navigation"
            >
              {isMenuOpen ? (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  ></path>
                </svg>
              ) : (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 6h16M4 12h16m-7 6h7"
                  ></path>
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div
          id="mobile-menu"
          className="lg:hidden bg-slate-900/95 backdrop-blur-md border-b border-[#003399]/50 shadow-lg transition-all duration-300"
        >
          <div className="px-6 py-4">
            <div className="flex flex-col space-y-2 mb-6">
              {navItems.map((item) => {
                const isActive = location === item.path;
                return (
                  <Link key={item.path} href={item.path}>
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => setIsMenuOpen(false)}
                      className={`relative w-full block text-left px-4 py-3 rounded-lg text-base font-semibold tracking-wide transition-all cursor-pointer text-white uppercase ${
                        isActive
                          ? `border border-[#0066DD]`
                          : ``
                      } focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[#0066DD]`}
                      data-testid={`mobile-link-${item.label.toLowerCase().replace(' ', '-')}`}
                    >
                      {isActive && (
                        <motion.div
                          layoutId="activeMobileNav"
                          className="absolute inset-0 bg-gradient-to-r from-[#0066DD]/60 to-[#003399]/60 rounded-lg border border-[#0066DD] shadow-lg shadow-[#0066DD]/50 -z-10"
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        />
                      )}
                      <span className="relative z-10">
                        {isActive ? (
                          <span className="text-white drop-shadow-[0_0_12px_rgba(0,102,221,0.6)]">
                            {item.label}
                          </span>
                        ) : (
                          item.label
                        )}
                      </span>
                    </motion.div>
                  </Link>
                );
              })}

              {/* mobile auth actions */}
              {!isAuthenticated ? (
                <button
                  onClick={() => {
                    setIsMenuOpen(false);
                    setLocation('/login');
                  }}
                  className="w-full text-left px-4 py-3 rounded-lg text-base font-semibold tracking-wide transition-all cursor-pointer text-white uppercase bg-white/5"
                  data-testid="mobile-button-signup-signin"
                >
                  Sign up / Sign in
                </button>
              ) : (
                <div className="flex flex-col gap-2">
                  <button
                    onClick={() => {
                      setIsMenuOpen(false);
                      setLocation('/account');
                    }}
                    className="w-full text-left px-4 py-3 rounded-lg text-base font-semibold transition-all cursor-pointer text-white uppercase bg-white/5"
                  >
                    Account
                  </button>
                </div>
              )}
            </div>

          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation;
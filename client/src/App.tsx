import { Switch, Route, useLocation  } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useEffect, useState, useRef } from "react";
import Home from "@/pages/Home";
import Login from "@/pages/Login";
import Register from "@/pages/register";
import Upload from "@/pages/Upload";
import Gallery from "@/pages/Gallery";
import Analyzing from "@/pages/Analyzing";
import History from "@/pages/history";
import Report from "@/pages/Report";
import Account from "@/pages/Account";
import NotFound from "@/pages/not-found";
import Preloader from "@/components/Preloader";
// Chatbot removed per request


function ScrollToTop() {
  const [location] = useLocation();
  
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location]);
  
  return null;
}

function Router() {
  return (
    <>
      <ScrollToTop />
        <Switch>
          {/* Routes: keep only the requested pages. Root remains register (signin/register) */}
          <Route path="/" component={Home} />
          <Route path="/home" component={Home} />
          <Route path="/login" component={Login} />
          <Route path="/register" component={Register} />
          <Route path="/upload" component={Upload} />
          <Route path="/gallery" component={Gallery} />
          <Route path="/analyzing" component={Analyzing} />
          <Route path="/history/:uid" component={History} />
          <Route path="/report" component={Report} />
          <Route path="/account" component={Account} />
          <Route component={NotFound} />
        </Switch>
     </>
  );
}


function App() {
  const preloaderInitRef = useRef(false);
  const [isPreloaderDone, setIsPreloaderDone] = useState(() => {
    try {
      return sessionStorage.getItem('preloader:shown') === 'true';
    } catch {
      return false;
    }
  });
  const [showPreloader, setShowPreloader] = useState(!isPreloaderDone);

  useEffect(() => {
    if (preloaderInitRef.current || isPreloaderDone) return;
    preloaderInitRef.current = true;

    const hideTimer = setTimeout(() => {
      setShowPreloader(false);
      try {
        sessionStorage.setItem('preloader:shown', 'true');
      } catch {}
      setTimeout(() => setIsPreloaderDone(true), 1000);
    }, 3000);

    return () => clearTimeout(hideTimer);
  }, [isPreloaderDone]);

  useEffect(() => {
    if (showPreloader) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
  }, [showPreloader]);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="dark">
          {!isPreloaderDone && <Preloader isLoading={showPreloader} />}
          <div
            style={{
              opacity: isPreloaderDone ? 1 : 0,
              transition: 'opacity 0.8s ease-in-out',
            }}
          >
            <Toaster />
            <Router />
          </div>
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;

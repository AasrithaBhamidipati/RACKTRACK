import Navigation from "@/components/Navigation";
import HeroSection from "@/components/HeroSection";
import AboutUs from "@/components/AboutUs";
import Features from "@/components/Features";
import WhyChooseUs from "@/components/WhyChooseUs";
import HowItWorks from "@/components/HowItWorks";
import Gallery from "@/components/Gallery";
import JoinToday from "@/components/JoinToday";
import Footer from "@/components/Footer";

import { FadeInUp } from "@/components/ScrollAnimation";
import SEO from "@/components/SEO";

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-background/95 dark">
      <SEO 
        title="RackTrack - AI-Powered Data Center Rack Auditing | 95% Faster Infrastructure Management"
        description="Automate your data center audits with RackTrack's AI-powered platform. Identify racks, switches, patch panels, ports, and cables in minutes - 95% faster than manual processes. Trusted by enterprise IT teams managing 100+ racks with 98%+ accuracy."
        ogTitle="RackTrack - Transform Data Center Auditing with AI"
        ogDescription="Cut rack audit time from 12 hours to 15 minutes. Computer vision and AI deliver professional documentation automatically - no manual tracing required."
      />
      <Navigation />
      <HeroSection />

      <FadeInUp>
        <AboutUs />
      </FadeInUp>
      <FadeInUp>
        <Features />
      </FadeInUp>
      <FadeInUp>
        <WhyChooseUs />
      </FadeInUp>
      <FadeInUp>
        <HowItWorks />
      </FadeInUp>
      <FadeInUp>
        <Gallery />
      </FadeInUp>
      <FadeInUp>
        <JoinToday />
      </FadeInUp>
      <Footer />
    </div>
  );
};

export default Home;
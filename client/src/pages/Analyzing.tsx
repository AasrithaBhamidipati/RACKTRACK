import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import Navigation from "@/components/Navigation";
import { useLocation } from "wouter";
import { Loader2, CheckCircle2, Server, Cable, Wifi, Network, Router } from "lucide-react";
import { getAuthHeaders } from "@/lib/auth";
import { getAnalyzingFilesById } from "@/lib/idb";
import darkGradientBg from "@assets/Screenshot 2025-09-25 162657_1759225397574.png";
import networkBg from "@assets/structural-connection-information-tech-coverage-generate-ai_98402-24492_1759225397574.jpg";

// Helper to convert base64 back to File
function dataURLtoFile(dataUrl: string, fileName: string, fileType: string, lastModified: number) {
  const arr = dataUrl.split(",");
  const match = arr[0].match(/:(.*?);/);
  const mime = match ? match[1] : fileType;
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], fileName, { type: mime, lastModified });
}


export default function Analyzing() {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<"processing" | "complete">("processing");
  const [, setLocation] = useLocation();

  useEffect(() => {
    (async () => {
      if (localStorage.getItem("isAuthenticated") !== "true") {
        setLocation("/login");
        return;
      }

      // Get files id and uploadType from sessionStorage
      const filesDataId = sessionStorage.getItem("analyzing_files_id");
      const uploadType = sessionStorage.getItem("analyzing_uploadType");
      if (!filesDataId || !uploadType) {
        setLocation("/upload");
        return;
      }

      try {
        const filesData = await getAnalyzingFilesById(filesDataId);
        if (!filesData) {
          setLocation("/upload");
          return;
        }

        const files = (filesData as any[]).map((fd: any) => dataURLtoFile(fd.data, fd.name, fd.type, fd.lastModified));
        const formData = new FormData();
        files.forEach((file: File) => formData.append("files", file));
        formData.append("uploadType", uploadType);

        // Simulate progress
        setProgress(10);
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
          headers: getAuthHeaders(),
        });
        setProgress(70);
        if (!res.ok) throw new Error("Upload failed");
        await res.json();
        setProgress(100);
        setStatus("complete");
        setTimeout(() => setLocation("/gallery"), 1200);
      } catch (err) {
        setStatus("processing");
        setProgress(0);
        setLocation("/upload");
      }
    })();
  }, [setLocation]);

  return (
    <div className="min-h-screen relative overflow-hidden pt-24">
      {/* Background with combined images */}
      <div className="absolute inset-0">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${darkGradientBg})` }}
        />
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-30"
          style={{ backgroundImage: `url(${networkBg})` }}
        />
        <div className="absolute inset-0 bg-black/70" />
      </div>

      {/* Floating decorative icons */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[15%] left-[8%] animate-pulse opacity-20">
          <div className="relative">
            <div className="absolute inset-0 bg-violet-500/30 blur-2xl"></div>
            <Server className="relative w-14 h-14 text-violet-400" />
          </div>
        </div>
        
        <div className="absolute top-[22%] right-[12%] animate-pulse opacity-15" style={{ animationDelay: '0.5s' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-blue-500/30 blur-2xl"></div>
            <Router className="relative w-16 h-16 text-blue-400" />
          </div>
        </div>
        
        <div className="absolute top-[45%] left-[5%] animate-pulse opacity-12" style={{ animationDelay: '1s' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-violet-500/30 blur-xl"></div>
            <Cable className="relative w-10 h-10 text-violet-400 rotate-45" />
          </div>
        </div>
        
        <div className="absolute top-[52%] right-[7%] animate-pulse opacity-15" style={{ animationDelay: '1.5s' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-purple-500/30 blur-xl"></div>
            <Network className="relative w-12 h-12 text-purple-400" />
          </div>
        </div>
        
        <div className="absolute bottom-[18%] left-[15%] animate-pulse opacity-18" style={{ animationDelay: '2s' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-blue-500/30 blur-2xl"></div>
            <Wifi className="relative w-14 h-14 text-blue-400" />
          </div>
        </div>
        
        <div className="absolute bottom-[25%] right-[18%] animate-pulse opacity-15" style={{ animationDelay: '2.5s' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-violet-500/30 blur-xl"></div>
            <Server className="relative w-10 h-10 text-violet-400" />
          </div>
        </div>

        <div className="absolute top-[35%] left-[20%] animate-pulse opacity-10" style={{ animationDelay: '3s' }}>
          <Cable className="w-8 h-8 text-blue-400 -rotate-12" />
        </div>
        
        <div className="absolute top-[38%] right-[25%] animate-pulse opacity-10" style={{ animationDelay: '3.5s' }}>
          <Cable className="w-8 h-8 text-violet-400 rotate-90" />
        </div>
      </div>

      <Navigation />
      <div className="relative flex items-center justify-center min-h-[calc(100vh-80px)]">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
          <Card className="bg-slate-900/50 backdrop-blur-sm border-violet-500/20 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-blue-500/5"></div>
            
            <CardHeader className="relative text-center space-y-4 pb-8">
              <div className="flex justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-violet-500/30 blur-3xl"></div>
                  <div className="relative rounded-full bg-gradient-to-br from-violet-500/20 to-blue-500/20 p-6">
                    {status === "complete" ? (
                      <CheckCircle2 className="h-16 w-16 text-green-400" data-testid="icon-complete" />
                    ) : (
                      <Loader2 className="h-16 w-16 text-violet-400 animate-spin" data-testid="icon-processing" />
                    )}
                  </div>
                </div>
              </div>

              <CardTitle className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-purple-400 to-blue-400" data-testid="text-title">
                {status === "complete" ? "Processing Complete!" : "Analyzing Your Content"}
              </CardTitle>
              
              <p className="text-lg text-white/70" data-testid="text-description">
                {status === "complete" 
                  ? "Your files have been successfully processed" 
                  : "Please wait while we process your uploaded files..."}
              </p>
            </CardHeader>

            <CardContent className="relative space-y-6 pb-8">
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-white/70">Progress</span>
                  <span className="text-white font-semibold" data-testid="text-progress">{progress}%</span>
                </div>
                
                <Progress 
                  value={progress} 
                  className="h-3 bg-slate-800/50"
                  data-testid="progress-bar"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
                <div className="flex items-center gap-3 p-4 rounded-lg bg-slate-800/30 border border-slate-700/30">
                  <div className={`p-2 rounded-lg ${progress > 30 ? 'bg-violet-500/20' : 'bg-slate-700/30'} transition-colors duration-300`}>
                    <Server className={`h-5 w-5 ${progress > 30 ? 'text-violet-400' : 'text-slate-500'}`} />
                  </div>
                  <div>
                    <p className="text-xs text-white/50">Upload</p>
                    <p className={`text-sm font-medium ${progress > 30 ? 'text-white' : 'text-white/50'}`}>
                      {progress > 30 ? 'Complete' : 'Processing'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-4 rounded-lg bg-slate-800/30 border border-slate-700/30">
                  <div className={`p-2 rounded-lg ${progress > 60 ? 'bg-purple-500/20' : 'bg-slate-700/30'} transition-colors duration-300`}>
                    <Network className={`h-5 w-5 ${progress > 60 ? 'text-purple-400' : 'text-slate-500'}`} />
                  </div>
                  <div>
                    <p className="text-xs text-white/50">Analysis</p>
                    <p className={`text-sm font-medium ${progress > 60 ? 'text-white' : 'text-white/50'}`}>
                      {progress > 60 ? 'Complete' : progress > 30 ? 'Processing' : 'Pending'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-4 rounded-lg bg-slate-800/30 border border-slate-700/30">
                  <div className={`p-2 rounded-lg ${progress >= 100 ? 'bg-blue-500/20' : 'bg-slate-700/30'} transition-colors duration-300`}>
                    <CheckCircle2 className={`h-5 w-5 ${progress >= 100 ? 'text-blue-400' : 'text-slate-500'}`} />
                  </div>
                  <div>
                    <p className="text-xs text-white/50">Finalize</p>
                    <p className={`text-sm font-medium ${progress >= 100 ? 'text-white' : 'text-white/50'}`}>
                      {progress >= 100 ? 'Complete' : progress > 60 ? 'Processing' : 'Pending'}
                    </p>
                  </div>
                </div>
              </div>

              {status === "complete" && (
                <div className="text-center pt-4">
                  <p className="text-sm text-green-400 animate-pulse" data-testid="text-redirect">
                    Redirecting to dashboard...
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

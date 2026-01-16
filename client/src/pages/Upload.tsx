
import { useState, useCallback, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileDropZone } from "@/components/FileDropZone";
import Navigation from "@/components/Navigation";
import { useToast } from "@/hooks/use-toast";
import { useMutation } from "@tanstack/react-query";
import { Image, Images, Video, ArrowLeft, Upload as UploadIcon, Zap } from "lucide-react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { getAuthHeaders } from "@/lib/auth";
import { saveAnalyzingFiles } from "@/lib/idb";

type UploadType = "single-image" | "multiple-images" | "video";

interface UploadTypeOption {
  id: UploadType;
  title: string;
  description: string;
  icon: React.ElementType;
  acceptedFormats: string;
  maxFiles: number;
  accentColor: string;
}

const uploadOptions: UploadTypeOption[] = [
  {
    id: "single-image",
    title: "Single Image",
    description: "Upload one image for analysis",
    icon: Image,
    acceptedFormats: "image/jpeg,image/png,image/jpg,image/webp",
    maxFiles: 1,
    accentColor: "from-blue-500 to-cyan-400"
  },
  {
    id: "multiple-images",
    title: "Multiple Images",
    description: "Upload multiple images at once",
    icon: Images,
    acceptedFormats: "image/jpeg,image/png,image/jpg,image/webp",
    maxFiles: 20,
    accentColor: "from-blue-600 to-blue-400"
  },
  {
    id: "video",
    title: "Video",
    description: "Upload a video for analysis",
    icon: Video,
    acceptedFormats: "video/mp4,video/webm,video/quicktime",
    maxFiles: 1,
    accentColor: "from-cyan-500 to-blue-500"
  },
];

export default function Upload() {
  const [step, setStep] = useState<1 | 2>(1);
  const [selectedType, setSelectedType] = useState<UploadTypeOption | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  useEffect(() => {
    if (localStorage.getItem("isAuthenticated") !== "true") {
      setLocation("/");
    }
  }, [setLocation]);

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
        headers: getAuthHeaders(),
      });
      if (!response.ok) {
        throw new Error("Upload failed");
      }
      return await response.json();
    },
    onSuccess: () => {
      setLocation("/gallery");
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to upload files",
        variant: "destructive",
      });
    },
  });

  const handleTypeSelect = (option: UploadTypeOption) => {
    setSelectedType(option);
    setStep(2);
  };

  const handleBack = () => {
    setStep(1);
    setSelectedType(null);
    setSelectedFiles([]);
  };

  const handleFilesSelected = useCallback((files: File[]) => {
    setSelectedFiles(files);
  }, []);

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload",
        variant: "destructive",
      });
      return;
    }
    const filesArray = Array.from(selectedFiles);
    Promise.all(filesArray.map(file => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          resolve({
            name: file.name,
            type: file.type,
            lastModified: file.lastModified,
            data: e.target?.result
          });
        };
        reader.readAsDataURL(file);
      });
    })).then((filesData) => {
      // Save large payload to IndexedDB and store only the id in sessionStorage
      saveAnalyzingFiles(filesData as any).then((id) => {
        sessionStorage.setItem("analyzing_files_id", id);
        sessionStorage.setItem("analyzing_uploadType", selectedType?.id || "");
        setLocation("/analyzing");
      }).catch(() => {
        toast({ title: "Error", description: "Failed to prepare files for analysis", variant: "destructive" });
      });
    });
  };

  return (
    <div className="min-h-screen bg-black">
      <Navigation />
      <div className="relative z-10 flex items-center justify-center min-h-[calc(100vh-80px)] py-20 px-4 pt-28">
        <div className="max-w-6xl mx-auto w-full">
          <motion.div 
            className="mb-16 text-center"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2.5 px-5 py-2 rounded-full bg-blue-500/5 border border-blue-500/20 mb-8 shadow-lg shadow-blue-500/5">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-blue-300 text-sm font-semibold tracking-wide">AI-Powered Analysis</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight" data-testid="text-page-title">
              <span className="text-white">{step === 1 ? "Choose " : "Upload "}</span>
              <span className={`bg-gradient-to-r ${step === 2 && selectedType ? selectedType.accentColor : 'from-blue-400 to-cyan-400'} bg-clip-text text-transparent`}>
                {step === 1 ? "Upload Type" : "Your Files"}
              </span>
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto font-light" data-testid="text-page-description">
              {step === 1 ? "Select your preferred upload method to begin" : `Drop your ${selectedType?.title.toLowerCase()} to start processing`}
            </p>

            {step === 1 && (
              <div className="flex items-center justify-center gap-2 mt-8">
                {[1, 2].map((s) => (
                  <div
                    key={s}
                    className={`h-1 rounded-full transition-all duration-300 ${
                      s === step ? 'w-12 bg-blue-500 shadow-lg shadow-blue-500/50' : 'w-1 bg-gray-700'
                    }`}
                  />
                ))}
              </div>
            )}
          </motion.div>

          <AnimatePresence mode="wait">
            {step === 1 ? (
              <motion.div 
                key="step1"
                className="grid grid-cols-1 md:grid-cols-3 gap-6" 
                data-testid="upload-type-selection"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4 }}
              >
                {uploadOptions.map((option, index) => {
                  const Icon = option.icon;

                  return (
                    <motion.div
                      key={option.id}
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    >
                      <Card
                        className="relative bg-black border border-gray-800 hover:border-blue-500/50 cursor-pointer transition-all duration-500 overflow-hidden group shadow-2xl hover:shadow-blue-500/20"
                        onClick={() => handleTypeSelect(option)}
                        data-testid={`card-type-${option.id}`}
                      >
                        <div className={`absolute inset-0 bg-gradient-to-br ${option.accentColor} opacity-0 group-hover:opacity-[0.03] transition-opacity duration-500`} />
                        <div className="absolute inset-0 bg-gradient-to-b from-blue-500/0 via-transparent to-blue-500/0 group-hover:from-blue-500/5 group-hover:to-blue-500/5 transition-all duration-500" />

                        <CardContent className="p-10">
                          <div className="flex flex-col items-center text-center gap-6">
                            <div className="relative">
                              <div className={`absolute inset-0 bg-gradient-to-br ${option.accentColor} opacity-20 blur-xl group-hover:opacity-40 transition-opacity duration-500`} />
                              <div className={`relative w-20 h-20 rounded-2xl bg-gradient-to-br ${option.accentColor} p-[1px] group-hover:scale-110 transition-transform duration-500`}>
                                <div className="w-full h-full rounded-2xl bg-black flex items-center justify-center">
                                  <Icon className="w-9 h-9 text-blue-400 group-hover:text-blue-300 transition-colors duration-300" strokeWidth={1.5} />
                                </div>
                              </div>
                            </div>
                            <div>
                              <h3 className="text-2xl font-bold text-white mb-3 tracking-tight" data-testid={`text-type-title-${option.id}`}>
                                {option.title}
                              </h3>
                              <p className="text-gray-500 text-sm mb-5 font-light">
                                {option.description}
                              </p>
                              <div className="flex flex-col gap-2 text-xs text-gray-600">
                                <div className="flex items-center justify-center gap-2">
                                  <span className={`w-1 h-1 rounded-full bg-gradient-to-r ${option.accentColor}`} />
                                  <span data-testid={`text-type-formats-${option.id}`} className="font-medium">
                                    {option.acceptedFormats.split(",").map(f => f.split("/")[1].toUpperCase()).join(", ")}
                                  </span>
                                </div>
                                <div className="flex items-center justify-center gap-2">
                                  <span className={`w-1 h-1 rounded-full bg-gradient-to-r ${option.accentColor}`} />
                                  <span className="font-medium">Max {option.maxFiles} {option.maxFiles === 1 ? 'file' : 'files'}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </motion.div>
            ) : (
              <motion.div 
                key="step2"
                className="space-y-8 max-w-4xl mx-auto" 
                data-testid="upload-interface"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4 }}
              >
                <Card className="bg-black border border-gray-800 overflow-hidden shadow-2xl">
                  <CardContent className="p-10">
                    <FileDropZone
                      onFilesSelected={handleFilesSelected}
                      acceptedTypes={selectedType?.acceptedFormats || ""}
                      maxFiles={selectedType?.maxFiles || 1}
                      uploadType={selectedType?.id || ""}
                      accentColor={selectedType?.accentColor || "from-blue-500 to-cyan-400"}
                    />
                  </CardContent>
                </Card>

                <div className="flex justify-between items-center gap-6">
                  <Button
                    variant="outline"
                    onClick={handleBack}
                    className="border-gray-800 bg-black text-gray-400 hover:bg-gray-900 hover:text-white hover:border-gray-700 gap-2 px-6 py-6 text-base font-semibold transition-all duration-300"
                    data-testid="button-back"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                  </Button>
                  <AnimatePresence>
                    {selectedFiles.length > 0 && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ duration: 0.3 }}
                      >
                        <Button
                          onClick={handleUpload}
                          className={`bg-gradient-to-r ${selectedType?.accentColor || 'from-blue-500 to-cyan-400'} hover:opacity-90 text-white font-bold px-10 py-6 text-base shadow-2xl shadow-blue-500/30 gap-2.5 transition-all duration-300`}
                          disabled={uploadMutation.isPending}
                          data-testid="button-upload"
                        >
                          {uploadMutation.isPending ? (
                            <>
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              >
                                <Zap className="w-5 h-5" />
                              </motion.div>
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <UploadIcon className="w-5 h-5" />
                              Start Analysis
                            </>
                          )}
                        </Button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {selectedFiles.length > 0 && (
                  <motion.div 
                    className="text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <p className="text-gray-600 text-sm font-light">
                      {selectedFiles.length} {selectedFiles.length === 1 ? 'file' : 'files'} ready for analysis
                    </p>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

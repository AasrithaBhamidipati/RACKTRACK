import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import Navigation from "@/components/Navigation";
import { getAuthHeaders } from "@/lib/auth";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Cable,
  Box,
  Grid3x3,
  Boxes,
  CheckCircle2,
  XCircle,
  ChevronLeft,
  ImageIcon,
  FileText,
  Loader2,
  Eye,
  Image as ImageIconAlt,
  Layers,
  Info,
} from "lucide-react";

interface FolderOption {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  color: string;
  aspectRatio: "horizontal" | "square" | "vertical";
}

const folderOptions: FolderOption[] = [
  {
    id: "cables",
    title: "Cables",
    description: "Cable segments",
    icon: Cable,
    color: "text-purple-600 dark:text-purple-400",
    aspectRatio: "square",
  },
  {
    id: "rack",
    title: "Rack",
    description: "Rack segments",
    icon: Box,
    color: "text-blue-600 dark:text-blue-400",
    aspectRatio: "vertical",
  },
  {
    id: "patch_panel",
    title: "Patch Panel",
    description: "Panel segments",
    icon: Grid3x3,
    color: "text-cyan-600 dark:text-cyan-400",
    aspectRatio: "square",
  },
  {
    id: "switch",
    title: "Switch",
    description: "Switch segments",
    icon: Boxes,
    color: "text-indigo-600 dark:text-indigo-400",
    aspectRatio: "horizontal",
  },
  {
    id: "connected_port",
    title: "Connected",
    description: "Active ports",
    icon: CheckCircle2,
    color: "text-emerald-600 dark:text-emerald-400",
    aspectRatio: "square",
  },
  {
    id: "empty_port",
    title: "Empty",
    description: "Inactive ports",
    icon: XCircle,
    color: "text-orange-600 dark:text-orange-400",
    aspectRatio: "square",
  },
];

interface SegmentImage {
  path: string;
  name: string;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
  confidence: number;
  class_name: string;
}

export default function Gallery() {
  const [, setLocation] = useLocation();
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportExists, setReportExists] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [reportLogs, setReportLogs] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<string>("name-asc");
  const [viewerImage, setViewerImage] = useState<SegmentImage | null>(null);
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{
    w: number;
    h: number;
  } | null>(null);
  const [originalImageName, setOriginalImageName] = useState<string | null>(
    null,
  );
  const { toast } = useToast();

  useEffect(() => {
    if (localStorage.getItem("isAuthenticated") !== "true") {
      setLocation("/login");
    }
  }, [setLocation]);

  useEffect(() => {
    const checkReportExists = async () => {
      try {
        const response = await fetch("/api/report/pdf", { method: "HEAD", headers: getAuthHeaders() });
        setReportExists(response.ok);
      } catch (error) {
        setReportExists(false);
      }
    };
    checkReportExists();
  }, []);

  const {
    data: images,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useQuery<SegmentImage[]>({
    queryKey: ["/api/segments", selectedFolder],
    queryFn: async ({ queryKey }) => {
      const response = await fetch(`${queryKey[0]}/${queryKey[1]}`, { headers: getAuthHeaders() });
      if (!response.ok) {
        throw new Error("Failed to fetch images");
      }
      return response.json();
    },
    enabled: !!selectedFolder,
    refetchOnMount: true,
    staleTime: 0,
  });

  const generateReportMutation = useMutation({
    mutationFn: async () => {
      console.log("=== REPORT GENERATION STARTED ===");
      console.log("Timestamp:", new Date().toISOString());

      // get current job id (useful to ensure server checks the right job folder)
      let jobId: string | null = null;
      try {
        const jobRes = await apiRequest("GET", "/api/current-job");
        const jobJson = await jobRes.json();
        jobId = jobJson?.job?.jobId || null;
      } catch (e) {
        console.warn("Failed to fetch current job before run-summary", e);
      }

      const response = await apiRequest("POST", "/api/run-summary", jobId ? { jobId } : undefined);
      const data = await response.json();

      console.log("=== REPORT GENERATION RESPONSE ===");
      console.log("Success:", data.success);
      console.log("Message:", data.message);

      // if the run-summary endpoint returns logs, save them for UI
      if (data.logs && Array.isArray(data.logs)) {
        const flat = data.logs.map((l: any) => {
          if (typeof l === "string") return l;
          if (l.type && l.text) return `[${l.type}] ${l.text}`;
          return JSON.stringify(l);
        });
        setReportLogs((prev) => prev.concat(flat));
      }

      // attach debug info if server returned it
      if (data.debug) {
        setReportLogs((prev) => prev.concat([`DEBUG: ${JSON.stringify(data.debug)}`]));
      }

      console.log("=== REPORT GENERATION COMPLETED ===\n");

      return data;
    },
    onSuccess: (data) => {
      console.log("✓ Report generation successful");
      toast({
        title: "Report generation started",
        description: "We are generating your report. It will be available here once ready.",
      });

      // Determine PDF URL to poll for readiness
      let url: string | null = null;
      if (data?.pdfUrl) {
        url = data.pdfUrl;
      } else if (data?.jobId) {
        url = `/api/report/${data.jobId}/pdf`;
      } else {
        url = "/api/report/pdf"; // fallback to current-job endpoint
      }

      setPdfUrl(url);
      setIsGenerating(true);
      setReportLogs([]);
      
      // If server returned summary text immediately, navigate to the report page to show it
      if (data?.summaryText && data?.jobId) {
        setTimeout(() => setLocation(`/report?reportId=${data.jobId}`), 200);
      }

      // Poll for PDF readiness
      let pollCount = 0;
      const pollInterval = 2500;
      const maxPolls = Math.floor(120000 / pollInterval);
      const poller = setInterval(async () => {
        pollCount += 1;
        try {
          const headRes = await fetch(url!, { method: "HEAD", headers: getAuthHeaders() });
          if (headRes.ok) {
            clearInterval(poller);
            setReportExists(true);
            setIsGenerating(false);
            toast({ title: "Report Ready", description: "Your PDF report is ready to view." });
          } else {
            // keep polling
          }
        } catch (err) {
          // ignore and continue polling
        }

        if (pollCount >= maxPolls) {
          clearInterval(poller);
          setIsGenerating(false);
          toast({ title: "Report Generation Timeout", description: "Report did not become available in time.", variant: "destructive" });
        }
      }, pollInterval);
    },
    onError: (error: Error) => {
      console.error("=== REPORT GENERATION FAILED ===");
      console.error("Error:", error.message);
      console.error("Stack:", error.stack);

      toast({
        title: "Generation Failed",
        description: error.message || "Failed to generate report",
        variant: "destructive",
      });
      setIsGenerating(false);
    },
  });

  const handleFolderClick = (folderId: string) => {
    if (selectedFolder === folderId) {
      setSelectedFolder(null);
    } else {
      setSelectedFolder(folderId);
    }
    setSearchQuery("");
  };

  const handleBack = () => {
    setLocation("/upload");
  };

  const handleGenerateReport = async () => {
    setReportExists(false);
    setIsGenerating(true);
    generateReportMutation.mutate();
  };

  const handleViewReport = () => {
    // Open inline PDF viewer instead of navigating away
    if (pdfUrl) {
      setShowPdfViewer(true);
    } else {
      setLocation("/report");
    }
  };

  const handleImageView = async (image: SegmentImage) => {
    setViewerImage(image);
    setBoundingBox(null);
    setOriginalImageName(null);

    try {
      const response = await fetch(
        `/api/segment-coordinates?path=${encodeURIComponent(image.path)}`,
        { headers: getAuthHeaders() },
      );
      if (response.ok) {
        const data = await response.json();
        if (data.coordinates) {
          setBoundingBox(data.coordinates);
          setOriginalImageName(data.originalImageName || null);
        }
      } else {
        setBoundingBox(null);
        setOriginalImageName(null);
      }
    } catch (error) {
      console.error("Error fetching coordinates:", error);
      setBoundingBox(null);
      setOriginalImageName(null);
    }
  };

  const getImageAspectClass = (
    aspectRatio: "horizontal" | "square" | "vertical",
  ) => {
    switch (aspectRatio) {
      case "horizontal":
        return "aspect-[16/9]";
      case "vertical":
        return "aspect-[9/16]";
      case "square":
      default:
        return "aspect-square";
    }
  };

  const getGridColsClass = (
    aspectRatio: "horizontal" | "square" | "vertical",
  ) => {
    switch (aspectRatio) {
      case "horizontal":
        return "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";
      case "vertical":
        return "grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6";
      case "square":
      default:
        return "grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5";
    }
  };

  const selectedFolderOption = folderOptions.find(
    (f) => f.id === selectedFolder,
  );

  const filteredImages = (
    images?.filter((img) =>
      img.name.toLowerCase().includes(searchQuery.toLowerCase()),
    ) || []
  ).sort((a, b) => {
    switch (sortBy) {
      case "name-asc":
        return a.name.localeCompare(b.name);
      case "name-desc":
        return b.name.localeCompare(a.name);
      default:
        return 0;
    }
  });

  const CurrentCategoryIcon = selectedFolderOption?.icon || ImageIconAlt;

  return (
    <div className="min-h-screen bg-background pt-24">
      <Navigation />

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleBack}
                data-testid="button-back"
                className="gap-2"
              >
                <ChevronLeft className="h-4 w-4" />
                Back to Upload
              </Button>
            </div>

            <div className="flex items-center gap-3">
              {reportExists && (
                <Button
                  variant="secondary"
                  size="default"
                  onClick={handleViewReport}
                  data-testid="button-view-report"
                  className="gap-2"
                >
                  <FileText className="h-4 w-4" />
                  View Report
                </Button>
              )}
              <Button
                variant="default"
                size="default"
                onClick={handleGenerateReport}
                disabled={isGenerating}
                data-testid="button-generate-report"
                className="gap-2"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <FileText className="h-4 w-4" />
                    {reportExists ? "Regenerate Report" : "Generate Report"}
                  </>
                )}
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-primary/10 rounded-lg">
              <ImageIconAlt className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1
                className="text-3xl font-bold text-foreground"
                data-testid="text-page-title"
              >
                Segmented Gallery
              </h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                Explore and analyze your network segmentation results
              </p>
            </div>
          </div>
        </div>

        {/* Progress indicator for report generation */}
        {isGenerating && (
          <Card className="mb-8">
            <CardContent className="p-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">
                    Analyzing and Generating Report
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Please wait...
                  </span>
                </div>
                <Progress
                  value={100}
                  className="h-2 animate-pulse"
                  data-testid="progress-report"
                />
                <p className="text-xs text-muted-foreground">
                  Running analysis scripts and generating PDF report
                </p>
                {reportLogs.length > 0 && (
                  <div className="mt-3 bg-gray-900/40 p-3 rounded text-xs font-mono overflow-auto max-h-40">
                    {reportLogs.map((l, i) => (
                      <div key={i} className="whitespace-pre-wrap text-white/80">{l}</div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Category Overview */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
              Segment Categories
            </h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {folderOptions.map((folder) => {
              const CategoryIcon = folder.icon;
              const isActive = selectedFolder === folder.id;

              return (
                <Card
                  key={folder.id}
                  className={`hover-elevate active-elevate-2 cursor-pointer transition-all duration-200 ${
                    isActive
                      ? "ring-2 ring-primary ring-offset-2 ring-offset-background"
                      : ""
                  }`}
                  onClick={() => handleFolderClick(folder.id)}
                  data-testid={`button-category-${folder.id}`}
                >
                  <CardContent className="p-5">
                    <div className="flex items-start justify-between mb-3">
                      <div
                        className={`p-2 bg-muted rounded-md ${folder.color}`}
                      >
                        <CategoryIcon className="h-5 w-5" />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-foreground mb-0.5">
                        {folder.title}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {folder.description}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Selected Category Content */}
        {selectedFolder ? (
          <div>
            {/* Category Header */}
            <div className="flex items-center gap-3 mb-6">
              <div
                className={`p-2.5 bg-muted rounded-lg ${selectedFolderOption?.color}`}
              >
                <CurrentCategoryIcon className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-foreground">
                  {selectedFolderOption?.title}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {isLoading
                    ? "Loading..."
                    : `${filteredImages.length} segment${filteredImages.length !== 1 ? "s" : ""} available`}
                </p>
              </div>
            </div>

            {/* Images Grid */}
            {isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {[...Array(10)].map((_, i) => (
                  <Card key={i} className="overflow-hidden">
                    <Skeleton className="aspect-square w-full" />
                  </Card>
                ))}
              </div>
            ) : error ? (
              <Card className="p-16 border-2 border-dashed">
                <div className="text-center">
                  <ImageIcon className="h-20 w-20 text-muted-foreground/30 mx-auto mb-6" />
                  <h3 className="text-2xl font-semibold mb-3">
                    Failed to Load Images
                  </h3>
                  <p className="text-muted-foreground text-lg mb-6">
                    There was an error fetching the segmented images
                  </p>
                  <Button
                    variant="secondary"
                    onClick={() => refetch()}
                    disabled={isFetching}
                    data-testid="button-retry"
                  >
                    {isFetching ? "Retrying..." : "Retry"}
                  </Button>
                </div>
              </Card>
            ) : filteredImages.length > 0 ? (
              <div
                className={`grid ${getGridColsClass(selectedFolderOption?.aspectRatio || "square")} gap-4`}
              >
                {filteredImages.map((image, index) => (
                  <Card
                    key={index}
                    className="group overflow-hidden cursor-pointer hover-elevate active-elevate-2 transition-all duration-200"
                    onClick={() => handleImageView(image)}
                    data-testid={`card-image-${index}`}
                  >
                    <div className="aspect-square relative bg-muted overflow-hidden">
                      <img
                        src={`/api/segments/image?path=${encodeURIComponent(image.path)}`}
                        alt={image.name}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                        data-testid={`img-segment-${index}`}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                      <div className="absolute top-2 right-2">
                        <Badge
                          variant="secondary"
                          className="bg-background/80 backdrop-blur-sm text-xs"
                        >
                          #{index + 1}
                        </Badge>
                      </div>
                      <div className="absolute bottom-2 left-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <div className="text-xs font-medium text-white bg-black/60 backdrop-blur-sm px-2 py-1 rounded-md truncate">
                          View Details
                        </div>
                      </div>
                    </div>
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <div
                            className="text-sm font-medium truncate text-foreground"
                            title={image.name}
                          >
                            {image.name}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="p-16 border-2 border-dashed">
                <div className="text-center">
                  <ImageIcon className="h-20 w-20 text-muted-foreground/30 mx-auto mb-6" />
                  <h3 className="text-2xl font-semibold mb-3">
                    {searchQuery ? "No matches found" : "No images found"}
                  </h3>
                  <p className="text-muted-foreground text-lg">
                    {searchQuery
                      ? "Try adjusting your search query"
                      : "Upload and process files to see segmented images here"}
                  </p>
                </div>
              </Card>
            )}
          </div>
        ) : (
          <Card
            className="p-16 border-2 border-dashed"
            data-testid="container-empty-state"
          >
            <div className="text-center">
              <div className="p-4 bg-muted/50 rounded-full mb-6 w-24 h-24 mx-auto flex items-center justify-center">
                <ImageIcon className="h-12 w-12 text-muted-foreground" />
              </div>
              <h3 className="text-2xl font-semibold mb-3">Select a Category</h3>
              <p className="text-muted-foreground text-lg max-w-md mx-auto">
                Choose a category from above to view your segmented network
                images
              </p>
            </div>
          </Card>
        )}
      </div>

      {/* Image Viewer Dialog */}
      <Dialog
        open={!!viewerImage}
        onOpenChange={() => {
          setViewerImage(null);
          setBoundingBox(null);
          setImgDimensions(null);
          setOriginalImageName(null);
        }}
      >
        <DialogContent className="max-w-7xl max-h-[95vh] overflow-hidden p-0">
          <div className="p-6 pb-4 border-b">
            <DialogHeader>
              <DialogTitle className="text-xl font-bold flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Layers className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <div className="text-foreground">{viewerImage?.name}</div>
                  <div className="text-sm font-normal text-muted-foreground mt-0.5">
                    {originalImageName ? "Detection View" : "Segment View"}
                  </div>
                </div>
              </DialogTitle>
            </DialogHeader>
          </div>

          <div className="flex flex-col lg:flex-row gap-6 p-6 overflow-y-auto max-h-[calc(95vh-8rem)]">
            {/* Main Image - Original with Detection */}
            <div className="flex-1 space-y-3">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-primary/60" />
                <h3 className="text-sm font-semibold text-foreground">
                  {originalImageName ? "Original Image with Detection" : "Full View"}
                </h3>
              </div>
              
              {originalImageName && boundingBox ? (
                <Card className="overflow-hidden p-4 bg-muted/50">
                  <div className="relative bg-background rounded-lg overflow-hidden">
                    <img
                      src={`/api/segments/image?path=${encodeURIComponent(originalImageName)}`}
                      alt="Original"
                      className="w-full h-auto"
                      onLoad={(e) => {
                        const img = e.target as HTMLImageElement;
                        setImgDimensions({
                          w: img.naturalWidth,
                          h: img.naturalHeight,
                        });
                      }}
                    />
                    {imgDimensions && (
                      <svg
                        className="absolute top-0 left-0 w-full h-full pointer-events-none"
                        viewBox={`0 0 ${imgDimensions.w} ${imgDimensions.h}`}
                        preserveAspectRatio="none"
                      >
                        <defs>
                          <filter id="glow">
                            <feGaussianBlur
                              stdDeviation="3"
                              result="coloredBlur"
                            />
                            <feMerge>
                              <feMergeNode in="coloredBlur" />
                              <feMergeNode in="SourceGraphic" />
                            </feMerge>
                          </filter>
                        </defs>
                        <rect
                          x={boundingBox.x1}
                          y={boundingBox.y1}
                          width={boundingBox.width}
                          height={boundingBox.height}
                          fill="none"
                          stroke="hsl(var(--primary))"
                          strokeWidth="4"
                          filter="url(#glow)"
                          className="animate-pulse"
                        />
                      </svg>
                    )}
                  </div>
                </Card>
              ) : (
                <Card className="overflow-hidden p-4 bg-muted/50">
                  <div className="relative bg-background rounded-lg overflow-hidden">
                    <img
                      src={`/api/segments/image?path=${encodeURIComponent(viewerImage?.path || "")}`}
                      alt={viewerImage?.name}
                      className="w-full h-auto"
                    />
                  </div>
                </Card>
              )}
            </div>

            {/* Isolated Segment - Smaller Card */}
            <div className="lg:w-80 space-y-3">
              <div className="flex items-center gap-2">
                <div className="p-1 bg-primary/10 rounded">
                  <Info className="h-3.5 w-3.5 text-primary" />
                </div>
                <h3 className="text-sm font-semibold text-foreground">
                  Isolated Segment
                </h3>
              </div>
              
              <Card className="overflow-hidden bg-muted/30">
                <div className="p-3">
                  <div className="bg-background rounded-lg p-4 flex items-center justify-center min-h-[200px]">
                    <img
                      src={`/api/segments/image?path=${encodeURIComponent(viewerImage?.path || "")}`}
                      alt={viewerImage?.name}
                      className="max-w-full h-auto object-contain"
                      style={{ maxHeight: '400px' }}
                    />
                  </div>
                </div>
                
                <div className="px-4 pb-4">
                  <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3 border">
                    <div className="flex items-start gap-2">
                      <Info className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                      <div>
                        This is the isolated segment extracted from the detection area shown in the original image.
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* PDF Viewer Dialog */}
      <Dialog open={showPdfViewer} onOpenChange={() => setShowPdfViewer(false)}>
        <DialogContent className="max-w-5xl max-h-[95vh] overflow-hidden p-0">
          <div className="p-4 border-b flex items-center justify-between">
            <DialogHeader>
              <DialogTitle className="text-lg font-bold">Analysis Report</DialogTitle>
            </DialogHeader>
            <div className="flex items-center gap-2">
              <Button variant="ghost" onClick={() => setShowPdfViewer(false)}>Close</Button>
              <Button
                variant="secondary"
                asChild
              >
                <a href={pdfUrl || "/api/report/pdf"} target="_blank" rel="noreferrer" download>
                  Download
                </a>
              </Button>
            </div>
          </div>

          <div className="p-4" style={{ height: '80vh' }}>
            {pdfUrl ? (
              <iframe
                src={pdfUrl}
                title="Analysis Report"
                className="w-full h-full border"
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader2 className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                  <p className="text-sm text-muted-foreground mt-2">Loading report... please wait</p>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

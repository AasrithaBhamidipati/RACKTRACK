
import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  ArrowLeft, 
  Download, 
  Trash2, 
  FileText, 
  Loader2,
  Calendar,
  Eye,
  Clock,
  Image as ImageIcon,
  Sparkles
} from "lucide-react";
import { getAuthHeaders } from "@/lib/auth";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { motion } from "framer-motion";

interface Report {
  id: string;
  userId: string;
  title: string;
  filename: string;
  pdfPath: string;
  processedImage?: string;
  createdAt: string;
}

export default function History() {
  const [, setLocation] = useLocation();
  const [reports, setReports] = useState<Array<{ report: Report; count: number }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  // Open PDF viewer for a specific report
  const handleViewPdf = async (reportId: string) => {
    setPdfLoading(true);
    setShowPdfViewer(true);
    try {
      const res = await fetch(`/api/report/${reportId}/pdf`, { headers: getAuthHeaders() });
      if (!res.ok) throw new Error(`Failed to fetch PDF: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setPdfUrl(url);
    } catch (err) {
      setError('Failed to open PDF');
      setPdfUrl(null);
    } finally {
      setPdfLoading(false);
    }
  };

  useEffect(() => {
    const fetchSession = async () => {
      try {
        const response = await fetch("/api/session", {
          headers: getAuthHeaders(),
        });
        if (response.ok) {
          const data = await response.json();
          const uid = data.user?.userId || data.user?.id;
          setUserId(uid);
          fetchHistory(uid);
        } else {
          setError("Not authenticated. Please log in.");
          setLoading(false);
        }
      } catch (err) {
        setError("Failed to verify session");
        setLoading(false);
      }
    };
    fetchSession();
  }, []);

  const fetchHistory = async (uid: string) => {
    try {
      const response = await fetch(`/api/history/${uid}`, {
        headers: getAuthHeaders(),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch history");
      }
      const data = await response.json();
      const incoming: Report[] = data.reports || [];
      const normalize = (p?: string | null) => {
        if (!p) return "__noimg__";
        return p.replace(/\\/g, "/").replace(/^[\.\/]+/, "");
      };

      const map = new Map<string, { report: Report; count: number }>();
      incoming.forEach((r) => {
        const key = r.processedImage ? normalize(r.processedImage) : r.pdfPath ? normalize(r.pdfPath) : r.id;
        const existing = map.get(key);
        if (!existing) {
          map.set(key, { report: r, count: 1 });
        } else {
          const a = new Date(existing.report.createdAt).getTime();
          const b = new Date(r.createdAt).getTime();
          if (b > a) {
            map.set(key, { report: r, count: existing.count + 1 });
          } else {
            map.set(key, { report: existing.report, count: existing.count + 1 });
          }
        }
      });

      const deduped = Array.from(map.values());
      deduped.sort((a, b) => new Date(b.report.createdAt).getTime() - new Date(a.report.createdAt).getTime());
      setReports(deduped);
      setLoading(false);
    } catch (err: any) {
      setError(err.message || "Failed to load history");
      setLoading(false);
    }
  };

  const handleDelete = async (reportId: string) => {
    setDeleting(reportId);
    try {
      const response = await fetch(`/api/reports/${reportId}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
      if (!response.ok) {
        throw new Error("Failed to delete report");
      }
      setReports((cur) => cur.filter((r) => r.report.id !== reportId));
    } catch (err: any) {
      setError(err.message || "Failed to delete report");
    } finally {
      setDeleting(null);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return {
      date: date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      }),
      time: date.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
  };

  const getPreviewImage = (r: Report) => {
    if (!r.processedImage) return null;
    const val = Array.isArray((r as any).processedImage) ? (r as any).processedImage[0] : String(r.processedImage);
    const separators = ["|", ",", ";"];
    for (const s of separators) {
      if (val.includes(s)) return val.split(s)[0];
    }
    return val;
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-2xl animate-pulse"></div>
            <Loader2 className="w-16 h-16 animate-spin text-blue-400 mx-auto relative" />
          </div>
          <p className="text-xl font-semibold text-white">Loading your reports...</p>
          <p className="text-sm text-white/60 mt-2">Please wait</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      {/* PDF Viewer Dialog (moved outside the report card map) */}
      <Dialog open={showPdfViewer} onOpenChange={(open) => {
        setShowPdfViewer(open);
        if (!open && pdfUrl) {
          URL.revokeObjectURL(pdfUrl);
          setPdfUrl(null);
        }
      }}>
        <DialogContent className="max-w-4xl max-h-[90vh] p-0">
          <div className="p-6 pb-4 border-b">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                Report PDF
              </DialogTitle>
            </DialogHeader>
          </div>
          <div className="overflow-auto max-h-[calc(90vh-5rem)]">
            {pdfLoading ? (
              <div className="flex items-center justify-center h-96">
                <Loader2 className="h-12 w-12 text-primary animate-spin" />
                <span className="ml-4 text-primary">Loading PDF...</span>
              </div>
            ) : pdfUrl ? (
              <iframe
                src={pdfUrl}
                className="w-full h-[calc(90vh-8rem)]"
                title="PDF Report"
              />
            ) : (
              <div className="flex items-center justify-center h-96 text-muted-foreground">No PDF available</div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative z-10">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={() => setLocation("/dashboard")}
                className="text-white/80 hover:text-white hover:bg-white/10 border border-white/10"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
              <Button
                variant="ghost"
                onClick={() => setLocation("/")}
                className="text-white/80 hover:text-white hover:bg-white/10 border border-white/10"
              >
                Home
              </Button>
            </div>
          </div>

          <div className="text-center max-w-3xl mx-auto">
            <Badge className="mb-4 bg-gradient-to-r from-blue-500/30 to-cyan-500/30 text-blue-100 border-blue-400/50 px-4 py-1.5">
              <Sparkles className="w-3 h-3 mr-1.5" />
              YOUR REPORTS
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Report{" "}
              <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                History
              </span>
            </h1>
            <p className="text-lg text-white/70">
              Access and manage all your generated infrastructure reports
            </p>
            {reports.length > 0 && (
              <div className="mt-4 inline-flex items-center gap-2 text-sm text-white/60">
                <FileText className="w-4 h-4" />
                <span>{reports.length} {reports.length === 1 ? 'report' : 'reports'} found</span>
              </div>
            )}
          </div>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Alert variant="destructive" className="mb-6 bg-red-500/10 border-red-500/50">
              <AlertDescription className="text-red-200">{error}</AlertDescription>
            </Alert>
          </motion.div>
        )}

        {reports.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <Card className="bg-white/5 border-white/10 backdrop-blur-sm">
              <CardContent className="py-20 text-center">
                <div className="relative inline-block mb-6">
                  <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-2xl"></div>
                  <div className="relative bg-gradient-to-br from-blue-500/20 to-cyan-500/20 p-6 rounded-full border border-blue-500/30">
                    <FileText className="w-12 h-12 text-blue-400" />
                  </div>
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">No Reports Yet</h3>
                <p className="text-white/60 mb-6 max-w-md mx-auto">
                  Start by processing an image to generate your first infrastructure report
                </p>
                <Button
                  onClick={() => setLocation("/upload")}
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white border-0"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  Create First Report
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {reports.map(({ report, count }, index) => {
              const preview = getPreviewImage(report);
              const previewUrl = preview ? `/api/segments/image?path=${encodeURIComponent(preview)}` : null;
              const { date, time } = formatDate(report.createdAt);

              return (
                <motion.div
                  key={report.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="group bg-white/5 border-white/10 backdrop-blur-sm hover:bg-white/10 hover:border-blue-500/50 transition-all duration-300 overflow-hidden h-full flex flex-col">
                    {/* Preview Image */}
                    <div className="relative h-48 bg-gradient-to-br from-slate-800 to-slate-900 overflow-hidden">
                      {previewUrl ? (
                        <button
                          onClick={async () => {
                            try {
                              const res = await fetch(`/api/report/${report.id}/pdf`, { headers: getAuthHeaders() });
                              if (!res.ok) throw new Error(`Failed to fetch PDF: ${res.status}`);
                              const blob = await res.blob();
                              const url = URL.createObjectURL(blob);
                              window.open(url, '_blank');
                              setTimeout(() => URL.revokeObjectURL(url), 2000);
                            } catch (err) {
                              console.error('Failed to open PDF:', err);
                              setError('Failed to open PDF');
                            }
                          }}
                          className="w-full h-full relative group/preview"
                        >
                          <img 
                            src={previewUrl} 
                            alt="preview" 
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover/preview:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                            <div className="bg-white/20 backdrop-blur-sm rounded-full p-3 border border-white/30">
                              <Eye className="w-6 h-6 text-white" />
                            </div>
                          </div>
                        </button>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <div className="text-center">
                            <ImageIcon className="w-12 h-12 text-white/30 mx-auto mb-2" />
                            <span className="text-xs text-white/50">No preview available</span>
                          </div>
                        </div>
                      )}
                      
                      {/* Badge overlay */}
                      {count > 1 && (
                        <div className="absolute top-3 right-3">
                          <Badge className="bg-blue-500/90 text-white border-0 backdrop-blur-sm">
                            {count}× generated
                          </Badge>
                        </div>
                      )}
                    </div>

                    {/* Content */}
                    <CardContent className="p-5 flex-1 flex flex-col">
                      <h3 className="text-lg font-bold text-white mb-3 group-hover:text-blue-400 transition-colors">
                        Merged Result
                      </h3>

                      <div className="space-y-2 mb-4 flex-1">
                        <div className="flex items-center gap-2 text-sm text-white/60">
                          <Calendar className="w-4 h-4 text-blue-400" />
                          <span>{date}</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-white/60">
                          <Clock className="w-4 h-4 text-cyan-400" />
                          <span>{time}</span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2 pt-4 border-t border-white/10">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="flex-1 hover:bg-blue-500/20 hover:text-blue-400 border border-white/10"
                          onClick={() => handleViewPdf(report.id)}
                        >
                          <Eye className="w-4 h-4 mr-1.5" />
                          View
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="flex-1 hover:bg-cyan-500/20 hover:text-cyan-400 border border-white/10"
                          onClick={async () => {
                            try {
                              const res = await fetch(`/api/report/${report.id}/pdf`, { headers: getAuthHeaders() });
                              if (!res.ok) throw new Error(`Failed to download PDF: ${res.status}`);
                              const blob = await res.blob();
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement('a');
                              a.href = url;
                              a.download = 'Merged_Result.pdf';
                              document.body.appendChild(a);
                              a.click();
                              document.body.removeChild(a);
                              setTimeout(() => URL.revokeObjectURL(url), 1000);
                            } catch (err) {
                              console.error('Download failed:', err);
                              setError('Download failed');
                            }
                          }}
                        >
                          <Download className="w-4 h-4 mr-1.5" />
                          Download
                        </Button>

                        <Button
                          variant="ghost"
                          size="sm"
                          className="hover:bg-red-500/20 hover:text-red-400 border border-white/10"
                          onClick={() => handleDelete(report.id)}
                          disabled={deleting === report.id}
                        >
                          {deleting === report.id ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Trash2 className="w-4 h-4" />
                          )}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

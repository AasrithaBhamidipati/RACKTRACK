import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
// Removed Tabs UI for a single-page Account layout
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { 
  ArrowLeft, 
  User as UserIcon, 
  Mail, 
  Calendar,
  Save,
  Loader2,
  Download,
  Trash2,
  FileText,
  Eye,
  Clock,
  Image as ImageIcon,
  Sparkles,
  Shield,
  LogOut,
  Edit,
  Camera,
  Check
} from "lucide-react";
import { getAuthHeaders, removeSessionId, getSessionId } from "@/lib/auth";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { updateProfileSchema, type UpdateProfile } from "@shared/schema";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { queryClient } from "@/lib/queryClient";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { User } from "@shared/schema";
import avatar1 from "@assets/generated_images/professional_woman_business.png";
import avatar2 from "@assets/generated_images/professional_man_casual.png";
import avatar3 from "@assets/generated_images/professional_woman_glasses.png";
import avatar4 from "@assets/generated_images/professional_executive_man.png";
import avatar5 from "@assets/generated_images/professional_young_man.png";
import avatar6 from "@assets/generated_images/professional_woman_dark_hair.png";
import avatar7 from "@assets/generated_images/professional_man_casual_wear.png";
import avatar8 from "@assets/generated_images/professional_blonde_woman.png";
import avatar9 from "@assets/generated_images/professional_man_friendly.png";
import avatar10 from "@assets/generated_images/professional_bearded_man.png";
import avatar11 from "@assets/generated_images/professional_unique_woman.png";
import avatar12 from "@assets/generated_images/professional_confident_man.png";

// Professional realistic anime avatars with cyan frames and tech backgrounds
const AVATAR_OPTIONS = [
  avatar1,
  avatar2,
  avatar3,
  avatar4,
  avatar5,
  avatar6,
  avatar7,
  avatar8,
  avatar9,
  avatar10,
  avatar11,
  avatar12
];

interface Report {
  id: string;
  userId: string;
  title: string;
  filename: string;
  pdfPath: string;
  processedImage?: string;
  createdAt: string;
}

export default function Account() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  // removed tab UI; render sections inline
  const [showAvatarPicker, setShowAvatarPicker] = useState(false);
  const [selectedAvatar, setSelectedAvatar] = useState<string>("");
  const [showReports, setShowReports] = useState(false);

  const { data: profileData, isLoading: profileLoading, refetch: refetchProfile } = useQuery<{ user: Omit<User, 'password'> & { profileImage?: string } }>({
    queryKey: ["/api/user/profile"],
  });

  const user = profileData?.user;

  const form = useForm<UpdateProfile & { profileImage?: string }>({
    resolver: zodResolver(updateProfileSchema),
    defaultValues: {
      email: "",
      fullName: "",
      profileImage: "",
    },
  });

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("profileImage") : null;
    if (user) {
      form.reset({
        email: user.email || "",
        fullName: user.fullName || "",
      });
      setSelectedAvatar((user as any).profileImage || saved || AVATAR_OPTIONS[0]);
    } else if (saved) {
      setSelectedAvatar(saved);
    }
  }, [user, form]);

  const updateProfileMutation = useMutation({
    mutationFn: async (data: UpdateProfile & { profileImage?: string }) => {
      const response = await fetch("/api/user/profile", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify({ ...data, profileImage: selectedAvatar }),
      });
      if (!response.ok) {
        throw new Error("Failed to update profile");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/user/profile"] });
      refetchProfile();
      // Trigger navigation update
      try {
        localStorage.setItem("profileImage", selectedAvatar);
      } catch (e) {}
      window.dispatchEvent(new Event("storage"));
      window.dispatchEvent(new Event("profile-updated"));
      toast({
        title: "Profile updated",
        description: "Your profile has been updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update profile",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: UpdateProfile) => {
    try {
      if (data.fullName) localStorage.setItem("username", data.fullName);
    } catch (e) {}
    updateProfileMutation.mutate({ ...data, profileImage: selectedAvatar });
  };

  const formatDate = (date: Date | string | null | undefined) => {
    if (!date) return "N/A";
    const d = new Date(date);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  if (profileLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 relative overflow-hidden">
      {/* Subtle animated background */}
      <div className="absolute inset-0 pointer-events-none">
        <motion.div
          className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 20, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.3, 0.5],
          }}
          transition={{ duration: 20, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <Button
            variant="ghost"
            className="text-white/60 hover:text-white hover:bg-white/5 mb-6"
            onClick={() => setLocation("/upload")}
            data-testid="button-back"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Go to Upload
          </Button>

          {/* Profile Header */}
          <Card className="bg-slate-900/50 border border-white/10 backdrop-blur-xl shadow-2xl">
            <CardContent className="p-8">
              <div className="flex flex-col md:flex-row items-center md:items-start gap-8">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 200 }}
                  className="relative"
                >
                  <Avatar className="w-36 h-36 border-4 border-slate-800 shadow-2xl">
                    {selectedAvatar && <AvatarImage src={selectedAvatar} alt="Profile" />}
                    <AvatarFallback className="bg-gradient-to-br from-blue-500 to-cyan-500 text-white text-5xl font-bold">
                      {user?.fullName?.charAt(0) || user?.username?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <Button
                    size="icon"
                    className="absolute -bottom-2 -right-2 rounded-full bg-blue-500 hover:bg-blue-600 shadow-xl"
                    onClick={() => setShowAvatarPicker(true)}
                    data-testid="button-change-avatar"
                  >
                    <Camera className="w-5 h-5" />
                  </Button>
                </motion.div>

                <div className="flex-1 text-center md:text-left">
                  <motion.h1
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="text-5xl font-bold text-white mb-3"
                  >
                    {user?.fullName || user?.username || "User"}
                  </motion.h1>
                  <p className="text-white/60 mb-4 flex items-center justify-center md:justify-start gap-2">
                    <Mail className="w-5 h-5" />
                    {user?.email || "No email set"}
                  </p>
                  <div className="flex flex-wrap gap-3 justify-center md:justify-start">
                    <Badge className="bg-blue-500/20 border-blue-500/40 text-blue-400 px-3 py-1">
                      <Shield className="w-4 h-4 mr-1.5" />
                      Verified Member
                    </Badge>
                    <Badge className="bg-white/10 border-white/20 text-white/80 px-3 py-1">
                      <Calendar className="w-4 h-4 mr-1.5" />
                      Joined {formatDate(user?.joinedAt)}
                    </Badge>
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <Button
                    onClick={() => {
                      const next = !showReports;
                      setShowReports(next);
                      if (next) {
                        setTimeout(() => {
                          const el = document.getElementById("account-reports-section");
                          if (el) el.scrollIntoView({ behavior: "smooth" });
                        }, 80);
                      }
                    }}
                    className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg w-full"
                    data-testid="button-view-reports"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    {showReports ? "Hide Reports" : "View Reports"}
                  </Button>
                  <Button
                    variant="destructive"
                    className="shadow-lg w-full"
                    onClick={() => {
                      try {
                        const sid = getSessionId();
                        if (sid)
                          fetch("/api/logout", {
                            method: "POST",
                            headers: { Authorization: `Bearer ${sid}` },
                          });
                      } catch (e) {}
                      removeSessionId();
                      try {
                        localStorage.removeItem("profileImage");
                      } catch (e) {}
                      localStorage.removeItem("isAuthenticated");
                      window.dispatchEvent(new Event("storage"));
                      window.dispatchEvent(new Event("profile-updated"));
                      window.location.href = "/";
                    }}
                    data-testid="button-logout-top"
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    Logout
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Avatar Picker Dialog */}
        <Dialog open={showAvatarPicker} onOpenChange={setShowAvatarPicker}>
          <DialogContent className="bg-slate-900 border-white/20 text-white max-w-2xl">
            <DialogHeader>
              <DialogTitle className="text-2xl">Choose Profile Picture</DialogTitle>
              <DialogDescription className="text-white/60">
                Select an avatar for your profile
              </DialogDescription>
            </DialogHeader>
            <div className="grid grid-cols-4 gap-4 py-4">
              {AVATAR_OPTIONS.map((avatar, idx) => (
                <motion.div
                  key={idx}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className={`relative cursor-pointer rounded-full overflow-hidden border-4 transition-all ${
                    selectedAvatar === avatar ? 'border-blue-500 shadow-lg shadow-blue-500/50' : 'border-white/10 hover:border-white/30'
                  }`}
                  onClick={() => {
                    setSelectedAvatar(avatar);
                    try {
                      localStorage.setItem("profileImage", avatar);
                      window.dispatchEvent(new Event("profile-updated"));
                    } catch (e) {
                      // ignore localStorage errors
                    }
                    setShowAvatarPicker(false);
                  }}
                  data-testid={`avatar-option-${idx}`}
                >
                  <img src={avatar} alt={`Avatar ${idx + 1}`} className="w-full h-full" />
                  {selectedAvatar === avatar && (
                    <div className="absolute inset-0 bg-blue-500/30 flex items-center justify-center backdrop-blur-sm">
                      <Check className="w-12 h-12 text-white drop-shadow-lg" />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </DialogContent>
        </Dialog>

        {/* Main Content: Show Edit Profile + Account Info side-by-side OR Reports */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          {!showReports ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Edit Profile Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="h-full bg-slate-900/50 border border-white/10 backdrop-blur-xl shadow-xl flex flex-col">
                  <CardHeader className="border-b border-white/10">
                    <CardTitle className="text-2xl text-white flex items-center gap-3">
                      <Edit className="w-6 h-6 text-blue-400" />
                      Edit Profile
                    </CardTitle>
                    <CardDescription className="text-white/60">
                      Update your personal information
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6 flex-1">
                    <Form {...form}>
                      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                        <FormField
                          control={form.control}
                          name="fullName"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-white/90 flex items-center gap-2">
                                <UserIcon className="w-4 h-4 text-blue-400" />
                                Full Name
                              </FormLabel>
                              <FormControl>
                                <Input
                                  {...field}
                                  placeholder="Enter your full name"
                                  className="h-12 bg-slate-800/50 border border-white/20 text-white placeholder:text-white/40 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20"
                                  data-testid="input-fullname"
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />

                        <FormField
                          control={form.control}
                          name="email"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-white/90 flex items-center gap-2">
                                <Mail className="w-4 h-4 text-blue-400" />
                                Email Address
                              </FormLabel>
                              <FormControl>
                                <Input
                                  {...field}
                                  type="email"
                                  placeholder="Enter your email"
                                  className="h-12 bg-slate-800/50 border border-white/20 text-white placeholder:text-white/40 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20"
                                  data-testid="input-email"
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />

                        <Button
                          type="submit"
                          disabled={updateProfileMutation.isPending}
                          className="w-full h-12 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold shadow-lg"
                          data-testid="button-save-profile"
                        >
                          {updateProfileMutation.isPending ? (
                            <>
                              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                              Saving Changes...
                            </>
                          ) : (
                            <>
                              <Save className="w-5 h-5 mr-2" />
                              Save Changes
                            </>
                          )}
                        </Button>
                      </form>
                    </Form>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Account Information Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="h-full bg-slate-900/50 border border-white/10 backdrop-blur-xl shadow-xl flex flex-col">
                  <CardHeader className="border-b border-white/10">
                    <CardTitle className="text-2xl text-white flex items-center gap-3">
                      <UserIcon className="w-6 h-6 text-blue-400" />
                      Account Information
                    </CardTitle>
                    <CardDescription className="text-white/60">
                      Your account details and settings
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6 space-y-6 flex-1">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between bg-white/5 rounded-md p-3">
                        <div className="text-sm text-white/70 flex items-center gap-2"><UserIcon className="w-4 h-4" /> Username</div>
                        <div className="text-white font-semibold" data-testid="text-username">{user?.username || "N/A"}</div>
                      </div>
                      <div className="flex items-center justify-between bg-white/5 rounded-md p-3">
                        <div className="text-sm text-white/70 flex items-center gap-2"><Mail className="w-4 h-4" /> Email Address</div>
                        <div className="text-white font-semibold truncate" data-testid="text-email">{user?.email || "Not set"}</div>
                      </div>
                      <div className="flex items-center justify-between bg-white/5 rounded-md p-3">
                        <div className="text-sm text-white/70 flex items-center gap-2"><UserIcon className="w-4 h-4" /> Full Name</div>
                        <div className="text-white font-semibold" data-testid="text-fullname">{user?.fullName || "Not set"}</div>
                      </div>
                      <div className="flex items-center justify-between bg-white/5 rounded-md p-3">
                        <div className="text-sm text-white/70 flex items-center gap-2"><Calendar className="w-4 h-4" /> Member Since</div>
                        <div className="text-white font-semibold" data-testid="text-joined-date">{formatDate(user?.joinedAt)}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          ) : (
            <div id="account-reports-section">
              <HistoryTab userId={user?.id} />
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}

function HistoryTab({ userId }: { userId?: string }) {
  const [, setLocation] = useLocation();
  const [reports, setReports] = useState<Array<{ report: Report; count: number }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [activeReportId, setActiveReportId] = useState<string | null>(null);
  const { toast } = useToast();
  // Open PDF viewer for a specific report
  const handleViewPdf = async (reportId: string) => {
    setPdfLoading(true);
    setShowPdfViewer(true);
    setActiveReportId(reportId);
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
    if (userId) {
      fetchHistory(userId);
    }
  }, [userId]);

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
      toast({
        title: "Report deleted",
        description: "The report has been removed successfully",
      });
    } catch (err: any) {
      toast({
        title: "Error",
        description: err.message || "Failed to delete report",
        variant: "destructive",
      });
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
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="bg-red-500/10 border-red-500/30">
        <AlertDescription className="text-red-400">{error}</AlertDescription>
      </Alert>
    );
  }

  if (reports.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <Card className="bg-slate-900/50 border border-white/10 backdrop-blur-xl shadow-2xl">
          <CardContent className="flex flex-col items-center justify-center py-20">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="mb-6"
            >
              <FileText className="w-20 h-20 text-blue-400/50" />
            </motion.div>
            <h3 className="text-3xl font-bold text-white mb-3">No Reports Yet</h3>
            <p className="text-white/60 mb-8 text-center max-w-md">
              Start your journey by uploading your first file and get instant AI-powered analysis
            </p>
            <Button
              onClick={() => setLocation("/upload")}
              className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg px-8 py-6 text-lg"
              data-testid="button-upload-now"
            >
              <Sparkles className="w-5 h-5 mr-2" />
              Upload Your First File
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <div className="space-y-4">
      {/* PDF Viewer Dialog (single instance) */}
      <Dialog open={showPdfViewer} onOpenChange={(open) => {
        setShowPdfViewer(open);
        if (!open && pdfUrl) {
          URL.revokeObjectURL(pdfUrl);
          setPdfUrl(null);
          setActiveReportId(null);
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

      {reports.map(({ report, count }) => {
        const { date, time } = formatDate(report.createdAt);
        const previewImage = getPreviewImage(report);

        return (
          <motion.div
            key={report.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="bg-slate-900/50 border border-white/10 backdrop-blur-xl hover:border-blue-400/30 hover:shadow-xl hover:shadow-blue-500/10 transition-all">
              <CardContent className="p-6">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="relative w-full md:w-56 h-56 flex-shrink-0 bg-slate-800/50 rounded-xl overflow-hidden border border-white/10">
                    {previewImage ? (
                      <img
                        src={`/api/segments/image?path=${encodeURIComponent(previewImage)}`}
                        alt={report.title}
                        className="w-full h-full object-cover"
                        data-testid={`img-report-${report.id}`}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <ImageIcon className="w-16 h-16 text-white/20" />
                      </div>
                    )}
                    {count > 1 && (
                      <Badge className="absolute top-3 right-3 bg-blue-500 text-white border-0 shadow-lg" data-testid={`badge-count-${report.id}`}>
                        <Sparkles className="w-3 h-3 mr-1" />
                        {count} versions
                      </Badge>
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-6">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-2xl font-bold text-white mb-3 truncate" data-testid={`text-title-${report.id}`}>
                          {report.title}
                        </h3>
                        <div className="flex flex-wrap gap-3 text-sm">
                          <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg border border-white/10">
                            <Calendar className="w-4 h-4 text-blue-400" />
                            <span className="text-white/80">{date}</span>
                          </div>
                          <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg border border-white/10">
                            <Clock className="w-4 h-4 text-blue-400" />
                            <span className="text-white/80">{time}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-3">
                      <Button
                        size="sm"
                        className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white shadow-lg"
                        onClick={() => handleViewPdf(report.id)}
                        data-testid={`button-view-${report.id}`}
                      >
                        <Eye className="w-4 h-4 mr-2" />
                        View Report
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-white/20 text-white hover:bg-white/10"
                        onClick={() => {
                          const link = document.createElement("a");
                          link.href = `/api/segments/pdf?path=${encodeURIComponent(report.pdfPath)}`;
                          link.download = report.filename;
                          link.click();
                        }}
                        data-testid={`button-download-${report.id}`}
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-red-400/30 text-red-400 hover:bg-red-500/10"
                        onClick={() => handleDelete(report.id)}
                        disabled={deleting === report.id}
                        data-testid={`button-delete-${report.id}`}
                      >
                        {deleting === report.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <>
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
}

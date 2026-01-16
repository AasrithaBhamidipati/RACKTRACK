import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { getAuthHeaders } from "@/lib/auth";
import { Download, ArrowLeft, Loader } from "lucide-react";
import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";

export default function Report() {
  const [, setLocation] = useLocation();
  const queryParams = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : new URLSearchParams();
  const viewerReportId = queryParams.get("reportId");

  const [pdfLoading, setPdfLoading] = useState(false);
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [summaryText, setSummaryText] = useState<string | null>(null);

  useEffect(() => {
    if (localStorage.getItem("isAuthenticated") !== "true") {
      setLocation("/login");
    }
  }, [setLocation]);

  useEffect(() => {
    // fetch summary text (if available)
    const fetchSummary = async () => {
      if (!viewerReportId) return;
      try {
        const res = await fetch(`/api/report/${viewerReportId}/summary`, { headers: getAuthHeaders() });
        if (res.ok) {
          const json = await res.json();
          if (json?.success && json.summary) setSummaryText(json.summary);
        }
      } catch (e) {
        console.debug("Failed to fetch summary text", e);
      }
    };
    fetchSummary();
  }, [viewerReportId]);

  // fetch the PDF only when the user requests it; opens in a new tab
  const fetchPdf = async () => {
    if (!viewerReportId) return;
    setPdfLoading(true);
    try {
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      while (true) {
        const dbgRes = await fetch(`/api/debug/job/${viewerReportId}`, { headers: getAuthHeaders(), credentials: "include" });
        if (dbgRes.ok) {
          const data = await dbgRes.json();
          if (data.pdfExists) {
            const endpoint = `/api/report/${viewerReportId}/pdf`;
            const res = await fetch(endpoint, { headers: getAuthHeaders(), credentials: "include" });
            if (res.ok) {
              const blob = await res.blob();
              const objectUrl = URL.createObjectURL(blob);
              window.open(objectUrl, '_blank');
              // optionally revoke after a delay
              setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
            }
            break;
          }
          if (data.job?.status === "failed") break;
        }
        await sleep(1000);
      }
    } catch (e) {
      console.debug("Failed to fetch PDF", e);
    } finally {
      setPdfLoading(false);
    }
  };

  // download the PDF when available
  const downloadPdf = async () => {
    if (!viewerReportId) return;
    setDownloadLoading(true);
    try {
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      while (true) {
        const dbgRes = await fetch(`/api/debug/job/${viewerReportId}`, { headers: getAuthHeaders(), credentials: "include" });
        if (dbgRes.ok) {
          const data = await dbgRes.json();
          if (data.pdfExists) {
            const endpoint = `/api/report/${viewerReportId}/pdf`;
            const res = await fetch(endpoint, { headers: getAuthHeaders(), credentials: "include" });
            if (res.ok) {
              const blob = await res.blob();
              const objectUrl = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = objectUrl;
              a.download = `Audit_Summary_Report_${viewerReportId || "report"}.pdf`;
              document.body.appendChild(a);
              a.click();
              a.remove();
              setTimeout(() => URL.revokeObjectURL(objectUrl), 30000);
            }
            break;
          }
          if (data.job?.status === "failed") break;
        }
        await sleep(1000);
      }
    } catch (e) {
      console.debug("Failed to download PDF", e);
    } finally {
      setDownloadLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Navigation />
      <div className="max-w-screen-xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              onClick={() => setLocation("/gallery")}
              className="gap-2"
            >
              <ArrowLeft className="h-4 w-4" strokeWidth={1.5} />
              Back
            </Button>
            <h2 className="text-2xl font-semibold">Audit Summary</h2>
          </div>

          <div className="flex items-center gap-2">
            {/* keep header compact; View Report button moved below the header */}
          </div>
        </div>

        {/* View Report button placed above the summary card as requested */}
        <div className="flex justify-end mb-4 gap-2">
          <Button onClick={downloadPdf} className="px-4 py-2" disabled={downloadLoading}>
            {downloadLoading ? <Loader className="h-4 w-4" /> : <><Download className="h-4 w-4 mr-2" />Download</>}
          </Button>
          <Button onClick={fetchPdf} className="px-4 py-2" disabled={pdfLoading}>
            {pdfLoading ? <Loader className="h-4 w-4" /> : "View Report"}
          </Button>
        </div>

        {summaryText && (
          <div className="mb-6 p-4 rounded-md bg-slate-900/40 border border-slate-800 text-slate-200">
            <h3 className="text-lg font-semibold mb-2">Summary</h3>
            <pre className="whitespace-pre-wrap text-sm leading-relaxed font-medium">{summaryText}</pre>
          </div>
        )}

      </div>

      <style>{`
        .h-screen-minus-header {
          height: calc(100vh - 200px);
          min-height: 70vh;
        }
      `}</style>
    </div>
  );
}
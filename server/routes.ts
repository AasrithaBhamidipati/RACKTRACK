import type { Express } from "express";
import express from "express";
import session from "express-session";
import createMemoryStore from "memorystore";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { pool } from "./db";
import { loginCredentialsSchema, insertUploadSchema, updateProfileSchema } from "@shared/schema";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { exec } from "child_process";
import { promisify } from "util";
import { spawn } from "child_process";
import { startJobWorker, getWorkerStatus } from "./jobWorker";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const execAsync = promisify(exec);

const filesDir = "files";
if (!fs.existsSync(filesDir)) {
  fs.mkdirSync(filesDir, { recursive: true });
}

const jobsOutputDir = path.join(process.cwd(), "jobs_output");
if (!fs.existsSync(jobsOutputDir)) {
  fs.mkdirSync(jobsOutputDir, { recursive: true });
}

const uploadTypeFolders = ["single-image", "multiple-images", "video"];
uploadTypeFolders.forEach((folder) => {
  const folderPath = path.join(filesDir, folder);
  if (!fs.existsSync(folderPath)) {
    fs.mkdirSync(folderPath, { recursive: true });
  }
});

const allowedTypes: Record<
  string,
  { mimes: string[]; extensions: string[]; maxFiles: number }
> = {
  "single-image": {
    mimes: ["image/jpeg", "image/png", "image/jpg", "image/webp"],
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
    maxFiles: 1,
  },
  "multiple-images": {
    mimes: ["image/jpeg", "image/png", "image/jpg", "image/webp"],
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
    maxFiles: 20,
  },
  video: {
    mimes: ["video/mp4", "video/webm", "video/quicktime"],
    extensions: [".mp4", ".webm", ".mov"],
    maxFiles: 1,
  },
};

const tempDir = path.join(filesDir, "temp");
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

const uploadStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, tempDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const basename = path.basename(file.originalname, ext);
    const safeName = basename.replace(/[^a-zA-Z0-9-_]/g, "_");
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${safeName}${ext}`);
  },
});

const upload = multer({
  storage: uploadStorage,
  limits: {
    fileSize: 50 * 1024 * 1024,
  },
});

declare module "express-session" {
  interface SessionData {
    userId?: string;
    username?: string;
    currentJobId?: string;
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  const MemoryStore = createMemoryStore(session);

  const sessionStore = new MemoryStore({
    checkPeriod: 86400000,
  });

  app.use(
    session({
      secret: process.env.SESSION_SECRET || "racktrack-secret-key-change-in-production",
      resave: false,
      saveUninitialized: false,
      store: sessionStore,
      cookie: {
        maxAge: 86400000,
        httpOnly: true,
        secure: false,
      },
    })
  );

  startJobWorker();

  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  app.post("/api/upload", upload.array("files", 20), async (req, res) => {
    try {
      const uploadType = req.body.uploadType;
      const files = req.files as Express.Multer.File[];

      if (!uploadType || !allowedTypes[uploadType]) {
        if (req.files) {
          (req.files as Express.Multer.File[]).forEach((file) => {
            if (fs.existsSync(file.path)) {
              fs.unlinkSync(file.path);
            }
          });
        }
        return res
          .status(400)
          .json({ success: false, message: "Invalid upload type" });
      }

      if (!files || files.length === 0) {
        return res.status(400).json({
          success: false,
          message: "No files uploaded or invalid file types",
        });
      }

      const typeConfig = allowedTypes[uploadType];

      const invalidFiles = files.filter((file) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const isMimeAllowed = typeConfig.mimes.includes(file.mimetype);
        const isExtAllowed = typeConfig.extensions.includes(ext);
        return !isMimeAllowed || !isExtAllowed;
      });

      if (invalidFiles.length > 0) {
        files.forEach((file) => fs.unlinkSync(file.path));
        return res.status(400).json({
          success: false,
          message:
            "Invalid file type(s). Please upload only allowed file formats.",
        });
      }

      if (files.length > typeConfig.maxFiles) {
        files.forEach((file) => fs.unlinkSync(file.path));
        return res.status(400).json({
          success: false,
          message: `Too many files. Maximum ${typeConfig.maxFiles} allowed for ${uploadType}`,
        });
      }

      const userId = req.session?.userId || null;
      const createdJobs: any[] = [];

      const jobId = `job_${Date.now()}_${randomUUID().slice(0, 8)}`;
      const jobInputDir = path.join(jobsOutputDir, jobId, "input");
      const jobOutputDir = path.join(jobsOutputDir, jobId, "output");

      fs.mkdirSync(jobInputDir, { recursive: true });
      fs.mkdirSync(jobOutputDir, { recursive: true });

      const uploadedFiles = await Promise.all(
        files.map(async (file) => {
          const fileName = path.basename(file.path);
          const newPath = path.join(jobInputDir, fileName);

          fs.renameSync(file.path, newPath);

          const uploadData = {
            fileName: file.originalname,
            fileType: file.mimetype,
            filePath: newPath,
            uploadType: uploadType,
          };

          const validatedData = insertUploadSchema.parse(uploadData);
          return await storage.createUpload(validatedData);
        }),
      );

      let inputPath: string;
      if (uploadType === "multiple-images") {
        inputPath = jobInputDir;
      } else {
        inputPath = uploadedFiles[0].filePath;
      }

      const firstUploadId = uploadedFiles[0]?.id || null;

      const job = await storage.createJob({
        jobId,
        userId: userId,
        uploadId: firstUploadId,
        jobType: "cpu",
        status: "waiting",
        inputPath: inputPath,
        outputPath: jobOutputDir,
      });

      await storage.addJobLog(jobId, `Job created for ${uploadedFiles.length} file(s)`);

      req.session.currentJobId = jobId;

      createdJobs.push({
        jobId: job.jobId,
        status: job.status,
        outputPath: jobOutputDir,
      });

      res.json({
        success: true,
        message: `Successfully uploaded ${uploadedFiles.length} file(s). Job queued for processing.`,
        uploads: uploadedFiles,
        jobs: createdJobs,
      });
    } catch (error) {
      console.error("Upload error:", error);
      if (req.files) {
        (req.files as Express.Multer.File[]).forEach((file) => {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
          }
        });
      }
      res.status(500).json({ success: false, message: "Upload failed" });
    }
  });

  app.post("/api/register", async (req, res) => {
    try {
      const { username, password } = req.body as { username?: string; password?: string };
      if (!username || !password) {
        return res.status(400).json({ success: false, message: "username and password are required" });
      }

      const existing = await storage.getUserByUsername(username);
      if (existing) {
        return res.status(400).json({ success: false, message: "Username already exists" });
      }

      const user = await storage.createUser({ username, password });
      res.status(201).json({ success: true, message: "User registered", user: { id: user.id, username: user.username } });
    } catch (err) {
      console.error("Register error:", err);
      res.status(500).json({ success: false, message: "Registration failed" });
    }
  });

  app.post("/api/login", async (req, res) => {
    try {
      const credentials = loginCredentialsSchema.parse(req.body);
      const isValid = await storage.validateCredentials(credentials);

      if (!isValid) {
        return res.status(401).json({ success: false, message: "Invalid credentials" });
      }

      const user = await storage.getUserByUsername(credentials.username);

      if (req.session) {
        req.session.userId = user?.id;
        req.session.username = credentials.username;
      }

      res.json({ 
        success: true, 
        message: "Login successful", 
        user: user ? { id: user.id, username: user.username } : undefined 
      });
    } catch (error) {
      res.status(400).json({ success: false, message: "Invalid request" });
    }
  });

  app.post("/api/logout", (req, res) => {
    if (req.session) {
      req.session.destroy(() => {});
    }
    res.json({ success: true, message: "Logged out" });
  });

  app.get("/api/session", (req, res) => {
    if (req.session?.userId) {
      return res.json({ success: true, user: { id: req.session.userId, username: req.session.username } });
    }
    return res.status(401).json({ success: false, message: "Not authenticated" });
  });

  app.get("/api/user/profile", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ success: false, message: "User not found" });
      }

      const { password, ...userWithoutPassword } = user;
      res.json({ success: true, user: userWithoutPassword });
    } catch (error) {
      console.error("Get profile error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch profile" });
    }
  });

  app.patch("/api/user/profile", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const profileData = updateProfileSchema.parse(req.body);
      const updatedUser = await storage.updateUserProfile(userId, { ...profileData, profileImage: req.body.profileImage });

      if (!updatedUser) {
        return res.status(404).json({ success: false, message: "User not found" });
      }

      const { password, ...userWithoutPassword } = updatedUser;
      res.json({ success: true, message: "Profile updated", user: userWithoutPassword });
    } catch (error) {
      console.error("Update profile error:", error);
      res.status(400).json({ success: false, message: "Failed to update profile" });
    }
  });

  app.get("/api/jobs", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const jobs = await storage.getUserJobs(userId);
      res.json({ success: true, jobs });
    } catch (error) {
      console.error("Get jobs error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch jobs" });
    }
  });

  app.get("/api/jobs/:jobId", async (req, res) => {
    try {
      const { jobId } = req.params;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      const logs = await storage.getJobLogs(jobId);

      res.json({ success: true, job, logs });
    } catch (error) {
      console.error("Get job error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch job" });
    }
  });

  app.get("/api/jobs/:jobId/output", async (req, res) => {
    try {
      const { jobId } = req.params;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      if (!job.outputPath || !fs.existsSync(job.outputPath)) {
        return res.status(404).json({ success: false, message: "Output not available yet" });
      }

      const files: { path: string; name: string; type: string }[] = [];

      const walkDir = (dir: string, basePath: string = ""): void => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relativePath = path.join(basePath, entry.name);

          if (entry.isDirectory()) {
            walkDir(fullPath, relativePath);
          } else {
            files.push({
              path: relativePath,
              name: entry.name,
              type: path.extname(entry.name).slice(1),
            });
          }
        }
      };

      walkDir(job.outputPath);

      res.json({ success: true, jobId, outputPath: job.outputPath, files });
    } catch (error) {
      console.error("Get job output error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch job output" });
    }
  });

  app.get("/api/worker/status", (req, res) => {
    const status = getWorkerStatus();
    res.json({ success: true, ...status });
  });

  app.get("/api/uploads", async (req, res) => {
    try {
      const uploads = await storage.getAllUploads();
      res.json({ success: true, uploads });
    } catch (error) {
      res.status(500).json({ success: false, message: "Failed to fetch uploads" });
    }
  });

  app.get("/api/segments/image", async (req, res) => {
    try {
      const imagePath = req.query.path as string;

      if (!imagePath) {
        return res.status(400).json({ success: false, message: "No image path provided" });
      }

      const resolvedPath = path.join(process.cwd(), imagePath);
      const segmentedOutputDir = path.join(process.cwd(), "segmented_output");
      const uploadsDir = path.join(process.cwd(), "uploads");
      const filesDir = path.join(process.cwd(), "files");
      const jobsDir = path.join(process.cwd(), "jobs_output");

      const isInSegmentedOutput = resolvedPath.startsWith(segmentedOutputDir);
      const isInUploads = resolvedPath.startsWith(uploadsDir);
      const isInFiles = resolvedPath.startsWith(filesDir);
      const isInJobs = resolvedPath.startsWith(jobsDir);

      if (!isInSegmentedOutput && !isInUploads && !isInFiles && !isInJobs) {
        return res.status(403).json({ success: false, message: "Access denied" });
      }

      if (!fs.existsSync(resolvedPath)) {
        return res.status(404).json({ success: false, message: "Image not found" });
      }

      res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, private");
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");

      res.sendFile(resolvedPath);
    } catch (error) {
      console.error("Error serving image:", error);
      res.status(500).json({ success: false, message: "Failed to serve image" });
    }
  });

  app.get("/api/segments/:folder", async (req, res) => {
    try {
      const folder = req.params.folder;
      const currentJobId = req.session.currentJobId;

      if (!currentJobId) {
        return res.json([]);
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.json([]);
      }

      // Python script outputs class folders directly to outputPath
      const outputDir = job.outputPath;

      if (!fs.existsSync(outputDir)) {
        return res.json([]);
      }

      const images: { path: string; name: string }[] = [];
      const imageExts = [".jpg", ".jpeg", ".png", ".webp"];

      const folderPath = path.join(outputDir, folder);
      if (fs.existsSync(folderPath) && fs.statSync(folderPath).isDirectory()) {
        const files = fs.readdirSync(folderPath);
        files.forEach((file) => {
          const ext = path.extname(file).toLowerCase();
          if (imageExts.includes(ext)) {
            const relativePath = path.join("jobs_output", currentJobId, "output", folder, file).replace(/\\/g, "/");
            images.push({
              path: relativePath,
              name: file,
            });
          }
        });
      }

      res.json(images);
    } catch (error) {
      console.error("Error fetching segments:", error);
      res.status(500).json({ success: false, message: "Failed to fetch segments" });
    }
  });

  // Helper to run report generation scripts for a specific job and wait for completion
  async function runReportForJob(job: any) {
    const scriptConfigs = [
      { script: "1_rack_match.py", inputSubfolder: "rack" },
      { script: "2_switch_match.py", inputSubfolder: "switch" },
      { script: "3_patchpanel_match.py", inputSubfolder: "patch_panel" },
      { script: "4_1_conneted_port_match.py", inputSubfolder: "connected_port" },
      { script: "4_port_match.py", inputSubfolder: null },
      { script: "5_cable_match.py", inputSubfolder: "cables" },
      { script: "6_merge_result.py", inputSubfolder: null },
    ];

    const jobResultsDir = path.join(job.outputPath, "Results");
    const segmentedOutputDir = job.outputPath;
    const pdfPath = path.join(jobResultsDir, "Merged_Result.pdf");

    const executionLogs: Array<{
      script: string;
      status: "success" | "error";
      stdout: string;
      stderr: string;
      error?: string;
      timestamp: string;
    }> = [];

    try {
      await storage.addJobLog(job.jobId, "Background report run started");

      for (const config of scriptConfigs) {
        const scriptPath = path.join(process.cwd(), "python_codes", config.script);
        const pythonExe = "python";
        let args: string[] = [];

        if (config.script === "6_merge_result.py") {
          args = [scriptPath, "--input", jobResultsDir, "--output", pdfPath];
        } else if (config.script === "4_port_match.py") {
          const inputDir = path.join(segmentedOutputDir, "empty_port");
          args = [scriptPath, "--input", inputDir, "--output", jobResultsDir, "--segmented-output", segmentedOutputDir];
        } else if (config.script === "4_1_conneted_port_match.py") {
          const inputDir = path.join(segmentedOutputDir, config.inputSubfolder!);
          args = [scriptPath, "--input", inputDir, "--output", jobResultsDir, "--segmented-output", segmentedOutputDir];
        } else {
          const inputDir = path.join(segmentedOutputDir, config.inputSubfolder!);
          args = [scriptPath, "--input", inputDir, "--output", jobResultsDir];
        }

        const timestamp = new Date().toISOString();
        await storage.addJobLog(job.jobId, `Running: ${config.script} ${args.join(" ")}`);
        console.log(`[Report Generation] Spawning: ${config.script} ${args.join(" ")}`);

        try {
          const child = spawn(pythonExe, args, { cwd: process.cwd() });
          let out = "";
          let err = "";

          child.stdout.on("data", (chunk) => {
            const s = chunk.toString();
            out += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.log(`[${config.script}]`, s);
          });
          child.stderr.on("data", (chunk) => {
            const s = chunk.toString();
            err += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.error(`[${config.script} Error]`, s);
          });

          const exitCode: number = await new Promise((resolve, reject) => {
            child.on("error", (e) => reject(e));
            child.on("close", (code) => resolve(code === null ? 1 : code));
          });

          if (exitCode !== 0) {
            executionLogs.push({ script: config.script, status: "error", stdout: out, stderr: err, error: `Exit code ${exitCode}`, timestamp });
            await storage.addJobLog(job.jobId, `Script ${config.script} failed with exit code ${exitCode}`);
            // Mark job failed and stop further processing
            await storage.updateJobStatus(job.jobId, "failed", `Script ${config.script} failed`);
            await storage.addJobLog(job.jobId, `Background report run failed`);
            return executionLogs;
          }

          executionLogs.push({ script: config.script, status: "success", stdout: out, stderr: err, timestamp });
          await storage.addJobLog(job.jobId, `${config.script} completed`);
        } catch (err: any) {
          console.error(`Error running ${config.script} in background:`, err);
          executionLogs.push({ script: config.script, status: "error", stdout: "", stderr: "", error: err.message || String(err), timestamp });
          await storage.updateJobStatus(job.jobId, "failed", `Script ${config.script} error`);
          await storage.addJobLog(job.jobId, `Background report run error: ${err.message || String(err)}`);
          return executionLogs;
        }
      }

      // Final check for PDF
      if (!fs.existsSync(pdfPath)) {
        await storage.addJobLog(job.jobId, "Merged PDF not found after background run");
        await storage.updateJobStatus(job.jobId, "failed", "Merged PDF not created");
        return executionLogs;
      }

      await storage.updateJobStatus(job.jobId, "done");
      await storage.addJobLog(job.jobId, "Background report generation completed successfully");
      console.log(`[Report Generation] Background run completed for job ${job.jobId}`);
      return executionLogs;
    } catch (bgErr: any) {
      console.error("Background report runner error:", bgErr);
      try { await storage.updateJobStatus(job.jobId, "failed", bgErr?.message || String(bgErr)); } catch (e) {}
      try { await storage.addJobLog(job.jobId, `Background runner error: ${bgErr?.message || String(bgErr)}`); } catch (e) {}
      return executionLogs;
    }
  }

  // Run the summary script (7_summary.py) against the current job's merged PDF
  app.post("/api/run-summary", async (req, res) => {
    try {
      // Allow explicit jobId to be supplied (useful for debugging); otherwise use session
      const suppliedJobId = (req.body && (req.body as any).jobId) || req.query?.jobId;
      const currentJobId = suppliedJobId || req.session.currentJobId;

      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found or has no output path" });
      }

      const mergedPdf = path.join(job.outputPath, "Results", "Merged_Result.pdf");
      const resultsDir = path.join(job.outputPath, "Results");
      const debug: any = { mergedPdfPath: mergedPdf, resultsDir };

      if (!fs.existsSync(mergedPdf)) {
        // If merged PDF isn't present yet, trigger the report generation and wait for it to complete
        await storage.addJobLog(currentJobId, "Merged PDF not found, triggering report generation before running summary");
        const genLogs = await runReportForJob(job);
        debug.genLogs = genLogs;
        // After running, re-check
        debug.mergedExists = fs.existsSync(mergedPdf);
        if (!debug.mergedExists) {
          const listing = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];
          debug.resultsListing = listing;
          return res.status(404).json({ success: false, message: "Merged PDF not found after generation attempt", debug });
        }
      } else {
        debug.mergedExists = true;
        debug.resultsListing = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir) : [];
      }

      const scriptPath = path.join(process.cwd(), "python_codes", "7_summary.py");
      const pythonExe = "python";

      const logs: Array<{ type: string; text: string }> = [];

      const child = spawn(pythonExe, [scriptPath, "--pdf", mergedPdf], { cwd: process.cwd() });

      child.stdout.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stdout", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.log("[summary stdout]", s);
      });

      child.stderr.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stderr", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.error("[summary stderr]", s);
      });

      const exitCode: number = await new Promise((resolve, reject) => {
        child.on("error", (e) => reject(e));
        child.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (exitCode !== 0) {
        await storage.addJobLog(currentJobId, `Summary script failed with exit code ${exitCode}`);
        await storage.updateJobStatus(currentJobId, "failed", `Summary script error`);
        return res.status(500).json({ success: false, message: "Summary script failed", exitCode, logs });
      }

      // after completion, check for the generated PDF (OUTPUT_PDF default is Audit_Summary_Report.pdf)
      const generatedPdfPath = path.join(process.cwd(), "Audit_Summary_Report.pdf");
      let pdfPathResponse = null;
      let summaryTextResponse: string | null = null;
      if (fs.existsSync(generatedPdfPath)) {
        // expose via jobs_output same folder: copy into job Results folder
        const dest = path.join(job.outputPath, "Results", path.basename(generatedPdfPath));
        try {
          fs.copyFileSync(generatedPdfPath, dest);
          pdfPathResponse = `/api/report/${job.jobId}/pdf`;
          // also try to copy/read a generated summary text file (same base name .txt)
          const generatedTxtPath = generatedPdfPath.replace(/\.pdf$/i, ".txt");
          if (fs.existsSync(generatedTxtPath)) {
            const destTxt = path.join(job.outputPath, "Results", path.basename(generatedTxtPath));
            try {
              fs.copyFileSync(generatedTxtPath, destTxt);
              const summaryText = fs.readFileSync(generatedTxtPath, { encoding: "utf-8" });
              summaryTextResponse = summaryText;
            } catch (e) {
              console.warn("Failed to copy/read generated summary text:", e);
            }
          }
        } catch (e) {
          console.warn("Failed to copy generated summary PDF to job folder:", e);
          pdfPathResponse = null;
        }
      }

      await storage.addJobLog(currentJobId, "Summary script completed");

      res.json({ success: true, message: "Summary completed", logs, pdfUrl: pdfPathResponse, summaryText: summaryTextResponse, jobId: job.jobId });
    } catch (error: any) {
      console.error("Run-summary error:", error);
      res.status(500).json({ success: false, message: error.message || "Run summary failed" });
    }
  });

  app.get("/api/current-job", async (req, res) => {
    try {
      const currentJobId = req.session.currentJobId;
      if (!currentJobId) {
        return res.json({ success: true, job: null });
      }
      const job = await storage.getJob(currentJobId);
      res.json({ success: true, job });
    } catch (error) {
      console.error("Error fetching current job:", error);
      res.status(500).json({ success: false, message: "Failed to fetch current job" });
    }
  });

  app.post("/api/generate-report", async (req, res) => {
    try {
      console.log("[Report Generation] Starting report generation process");

      const currentJobId = req.session.currentJobId;
      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found or has no output path" });
      }

      const jobResultsDir = path.join(job.outputPath, "Results");
      // Python script outputs class folders directly to outputPath
      const segmentedOutputDir = job.outputPath;

      if (!fs.existsSync(jobResultsDir)) {
        fs.mkdirSync(jobResultsDir, { recursive: true });
      }

      const pdfPath = path.join(jobResultsDir, "Merged_Result.pdf");

      // If a PDF already exists for this job and client didn't request a forced regen,
      // return immediately so the frontend can display it without waiting for regeneration.
      const forceRegen = req.query?.force === "true" || req.body?.force === true;
      if (fs.existsSync(pdfPath) && !forceRegen) {
        console.log("[Report Generation] PDF already exists; returning existing PDF info");
        const pdfUrl = `/api/report/${job.jobId}/pdf`;
        return res.json({
          success: true,
          message: "Report already generated",
          jobId: job.jobId,
          pdfPath: "Results/Merged_Result.pdf",
          pdfUrl,
        });
      }

      // If forcing regeneration, remove existing PDF so it will be recreated.
      if (fs.existsSync(pdfPath) && forceRegen) {
        try {
          fs.unlinkSync(pdfPath);
          console.log("[Report Generation] Cleared existing PDF (force)");
        } catch (e) {
          console.warn("[Report Generation] Failed to remove existing PDF:", e);
        }
      }

      const scriptConfigs = [
        { script: "1_rack_match.py", inputSubfolder: "rack" },
        { script: "2_switch_match.py", inputSubfolder: "switch" },
        { script: "3_patchpanel_match.py", inputSubfolder: "patch_panel" },
        { script: "4_1_conneted_port_match.py", inputSubfolder: "connected_port" },
        { script: "4_port_match.py", inputSubfolder: null },
        { script: "5_cable_match.py", inputSubfolder: "cables" },
        { script: "6_merge_result.py", inputSubfolder: null },
      ];

      const executionLogs: Array<{
        script: string;
        status: "success" | "error";
        stdout: string;
        stderr: string;
        error?: string;
        timestamp: string;
      }> = [];

      // Run scripts in background so the HTTP request doesn't block and trigger 504s.
      // Use the helper below to run the report generation asynchronously.
      runReportForJob(job).catch((err) => {
        console.error("Background runReportForJob error:", err);
      });

      // Immediately return job info so client can fetch the job-specific PDF endpoint
      const pdfUrl = `/api/report/${job.jobId}/pdf`;

      res.json({
        success: true,
        message: "Report generation started",
        jobId: job.jobId,
        pdfPath: "Results/Merged_Result.pdf",
        pdfUrl,
        logs: executionLogs,
      });
    } catch (error: any) {
      console.error("Report generation error:", error);
      res.status(500).json({
        success: false,
        message: error.message || "Report generation failed",
      });
    }
  });

  app.get("/api/report/pdf", async (req, res) => {
    try {
      const currentJobId = req.session.currentJobId;
      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found" });
      }

      const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "Report PDF not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", 'inline; filename="Merged_Result.pdf"');
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.get("/api/user/reports", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const reports = await storage.getUserReports(userId);
      res.json({ success: true, reports });
    } catch (error) {
      console.error("Get reports error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch reports" });
    }
  });

  app.get("/api/user/reports/:id/pdf", async (req, res) => {
    try {
      const reportId = req.params.id;
      const report = await storage.getReport(reportId);

      if (!report) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      const pdfPath = path.join(process.cwd(), report.pdfPath);

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "PDF file not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", `inline; filename="${report.filename}"`);
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving report PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.delete("/api/user/reports/:id", async (req, res) => {
    try {
      const reportId = req.params.id;
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const report = await storage.getReport(reportId);
      if (!report) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (report.userId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized to delete this report" });
      }

      const pdfPath = path.join(process.cwd(), report.pdfPath);
      if (fs.existsSync(pdfPath)) {
        fs.unlinkSync(pdfPath);
      }

      await storage.deleteReport(reportId);

      res.json({ success: true, message: "Report deleted successfully" });
    } catch (error) {
      console.error("Delete report error:", error);
      res.status(500).json({ success: false, message: "Failed to delete report" });
    }
  });

  app.get("/api/history/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const sessionUserId = req.session?.userId;

      if (!sessionUserId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      if (sessionUserId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized" });
      }

      const userJobs = await storage.getUserJobs(userId);
      
      const reports: Array<{
        id: string;
        userId: string;
        title: string;
        filename: string;
        pdfPath: string;
        processedImage?: string;
        createdAt: string;
      }> = [];

      for (const job of userJobs) {
        if (job.status === "done" && job.outputPath) {
          const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");
          
          if (fs.existsSync(pdfPath)) {
            let processedImage: string | undefined;
            const outputDir = job.outputPath;
            const classFolders = ["rack", "switch", "patch_panel", "cables", "connected_port", "empty_port"];
            
            for (const folder of classFolders) {
              const folderPath = path.join(outputDir, folder);
              if (fs.existsSync(folderPath)) {
                const files = fs.readdirSync(folderPath);
                const imageFile = files.find(f => /\.(jpg|jpeg|png|webp)$/i.test(f));
                if (imageFile) {
                  processedImage = path.join("jobs_output", job.jobId, "output", folder, imageFile).replace(/\\/g, "/");
                  break;
                }
              }
            }

            reports.push({
              id: job.jobId,
              userId: job.userId || userId,
              title: "Merged Result",
              filename: "Merged_Result.pdf",
              pdfPath: pdfPath,
              processedImage,
              createdAt: job.createdAt ? new Date(job.createdAt).toISOString() : new Date().toISOString(),
            });
          }
        }
      }

      res.json({ success: true, reports });
    } catch (error) {
      console.error("Get history error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch history" });
    }
  });

  app.get("/api/report/:id/pdf", async (req, res) => {
    try {
      const jobId = req.params.id;
      const sessionUserId = req.session?.userId;

      if (!sessionUserId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const job = await storage.getJob(jobId);

      if (!job || !job.outputPath) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (job.userId !== sessionUserId) {
        console.warn(`[PDF ACCESS DENIED] jobId=${jobId} job.userId=${job.userId} sessionUserId=${sessionUserId}`);
        return res.status(403).json({ success: false, message: "Not authorized to view this report" });
      }

      const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "PDF file not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", 'inline; filename="Merged_Result.pdf"');
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving report PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.get("/api/report/:id/summary", async (req, res) => {
    try {
      const jobId = req.params.id;
      const job = await storage.getJob(jobId);
      if (!job || !job.outputPath) return res.status(404).json({ success: false, message: "Job not found" });

      const txtPath = path.join(job.outputPath, "Results", "Audit_Summary_Report.txt");
      if (!fs.existsSync(txtPath)) {
        return res.status(404).json({ success: false, message: "Summary text not found" });
      }

      const content = fs.readFileSync(txtPath, { encoding: "utf-8" });
      res.json({ success: true, summary: content });
    } catch (err) {
      console.error("Error serving summary text:", err);
      res.status(500).json({ success: false, message: "Failed to serve summary" });
    }
  });

  // Debug endpoint: return job info and PDF existence for troubleshooting
  app.get("/api/debug/job/:id", async (req, res) => {
    try {
      const jobId = req.params.id;
      const sessionUserId = req.session?.userId || null;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      const pdfPath = job.outputPath ? path.join(job.outputPath, "Results", "Merged_Result.pdf") : null;
      const pdfExists = pdfPath ? fs.existsSync(pdfPath) : false;

      res.json({
        success: true,
        job: {
          jobId: job.jobId,
          userId: job.userId,
          status: job.status,
          inputPath: job.inputPath,
          outputPath: job.outputPath,
          createdAt: job.createdAt,
        },
        sessionUserId,
        pdfPath,
        pdfExists,
      });
    } catch (err) {
      console.error("Debug job endpoint error:", err);
      res.status(500).json({ success: false, message: "Debug endpoint error" });
    }
  });

  app.delete("/api/reports/:id", async (req, res) => {
    try {
      const jobId = req.params.id;
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const job = await storage.getJob(jobId);
      if (!job) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (job.userId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized to delete this report" });
      }

      if (job.outputPath) {
        const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");
        if (fs.existsSync(pdfPath)) {
          fs.unlinkSync(pdfPath);
        }
      }

      res.json({ success: true, message: "Report deleted successfully" });
    } catch (error) {
      console.error("Delete report error:", error);
      res.status(500).json({ success: false, message: "Failed to delete report" });
    }
  });

  const server = createServer(app);
  return server;
}

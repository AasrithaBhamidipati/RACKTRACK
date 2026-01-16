import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";
import { storage, type Report } from "./storage";
import type { Job } from "@shared/schema";
import { randomUUID } from "crypto";
import { db } from "./db";
import { sql } from "drizzle-orm";
import * as logger from "./logger";

// Default parallel workers: use number of CPU cores minus one for responsiveness
const DEFAULT_PARALLEL = Math.max(1, (process.env.MAX_PARALLEL_JOBS ? parseInt(process.env.MAX_PARALLEL_JOBS, 10) : (os.cpus() ? os.cpus().length - 1 : 2)));
const MAX_PARALLEL_JOBS = 10;
const POLL_INTERVAL_MS = 5000;
const DB_RETRY_INTERVAL_MS = 10000;
const MAX_DB_RETRIES = 30;

let runningJobs = 0;
let isRunning = false;
let dbConnected = false;

async function processPythonScript(
  scriptPath: string,
  inputPath: string,
  outputDir: string,
  jobId?: string
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve, reject) => {
    const scriptFullPath = path.isAbsolute(scriptPath)
      ? scriptPath
      : path.resolve(process.cwd(), scriptPath);
    const inputFullPath = path.isAbsolute(inputPath)
      ? inputPath
      : path.resolve(process.cwd(), inputPath);

    // Decide whether to call the script with flags or positional args.
    let spawnArgs: string[] = [];
    try {
      const scriptText = fs.readFileSync(scriptFullPath, "utf8");
      const usesFlagInput = /add_argument\([^\)]*["']--input["']|add_argument\([^\)]*["']-i["']/i.test(scriptText);
      if (usesFlagInput) {
        spawnArgs = [scriptFullPath, "--input", inputFullPath, "--output", outputDir];
        if (jobId) logger.streamToJob(jobId)(`Running: ${path.basename(scriptFullPath)} --input ${inputFullPath} --output ${outputDir}`);
        else logger.info(`Running: ${scriptFullPath} --input ${inputFullPath} --output ${outputDir}`, "job-runner");
      } else {
        spawnArgs = [scriptFullPath, inputFullPath, outputDir];
        if (jobId) logger.streamToJob(jobId)(`Running: ${path.basename(scriptFullPath)} ${inputFullPath} ${outputDir}`);
        else logger.info(`Running: ${scriptFullPath} ${inputFullPath} ${outputDir}`, "job-runner");
      }
    } catch (e) {
      // Fallback to positional args if reading fails
      spawnArgs = [scriptFullPath, inputFullPath, outputDir];
      if (jobId) logger.streamToJob(jobId)(`Running (fallback): ${path.basename(scriptFullPath)} ${inputFullPath} ${outputDir}`);
      else logger.info(`Running (fallback): ${scriptFullPath} ${inputFullPath} ${outputDir}`, "job-runner");
    }

    if (!fs.existsSync(scriptFullPath)) {
      return reject(new Error(`Python script not found: ${scriptFullPath}`));
    }
    if (!fs.existsSync(inputFullPath)) {
      return reject(new Error(`Input path not found: ${inputFullPath}`));
    }

    // If `outputDir` looks like a file path (has an extension), create its parent directory
    try {
      const looksLikeFile = path.extname(outputDir) !== "";
      if (looksLikeFile) {
        fs.mkdirSync(path.dirname(outputDir), { recursive: true });
      } else {
        fs.mkdirSync(outputDir, { recursive: true });
      }
    } catch (e) {
      // fallback to creating the path as-is if anything unexpected happens
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const child = spawn("python", spawnArgs, {
      windowsHide: true,
      cwd: process.cwd(),
    });

    let stdout = "";
    let stderr = "";

    if (jobId) {
      child.stdout.on("data", (chunk) => {
        const s = chunk.toString();
        stdout += s;
        logger.streamToJob(jobId)(s);
      });
      child.stderr.on("data", (chunk) => {
        const s = chunk.toString();
        stderr += s;
        logger.streamToJob(jobId)(s);
      });
    } else {
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
    }

    child.on("error", (err) => reject(err));

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr, code: code ?? 0 });
      } else {
        reject(new Error(`Python exited with code ${code}\n${stderr}`));
      }
    });
  });
}

async function processJob(job: Job): Promise<void> {
  const startTime = Date.now();
  logger.startJobGroup(job.jobId, "processing");

  try {
    await storage.updateJobStatus(job.jobId, "processing");
    await storage.addJobLog(job.jobId, "Job started processing");

    const inputPath = job.inputPath;
    const outputDir = job.outputPath || path.join(process.cwd(), "jobs_output", job.jobId);

    // If `outputDir` looks like a file path (has an extension), create its parent directory
    try {
      const looksLikeFile = path.extname(outputDir) !== "";
      if (looksLikeFile) {
        fs.mkdirSync(path.dirname(outputDir), { recursive: true });
      } else {
        fs.mkdirSync(outputDir, { recursive: true });
      }
    } catch (e) {
      // fallback to creating the path as-is if anything unexpected happens
      fs.mkdirSync(outputDir, { recursive: true });
    }

    let scriptPath = "";
    const uploadType = path.dirname(inputPath).split(path.sep).pop();

    if (uploadType === "single-image" || inputPath.match(/\.(jpg|jpeg|png|webp)$/i)) {
      scriptPath = path.join(process.cwd(), "python_codes", "single.py");
    } else if (uploadType === "multiple-images") {
      scriptPath = path.join(process.cwd(), "python_codes", "multii.py");
    } else if (uploadType === "video" || inputPath.match(/\.(mp4|webm|mov)$/i)) {
      scriptPath = path.join(process.cwd(), "python_codes", "video.py");
    } else {
      scriptPath = path.join(process.cwd(), "python_codes", "single.py");
    }

    await storage.addJobLog(job.jobId, `Running script: ${path.basename(scriptPath)}`);
    
    const result = await processPythonScript(scriptPath, inputPath, outputDir, job.jobId);
    
    await storage.addJobLog(job.jobId, `Script completed. stdout: ${result.stdout.slice(0, 500)}`);
    
    if (result.stderr) {
      await storage.addJobLog(job.jobId, `stderr: ${result.stderr.slice(0, 500)}`);
    }

    // Now generate the PDF report using the segmentation output
    await storage.addJobLog(job.jobId, "Starting PDF report generation...");
    logger.info(`Starting PDF generation for job ${job.jobId}`, "JobWorker");

    // Determine which scripts to run based on upload type
    const reportScripts = [
      { script: "1_rack_match.py", input: "rack" },
      { script: "2_switch_match.py", input: "switch" },
      { script: "3_patchpanel_match.py", input: "patch_panel" },
      { script: "4_1_conneted_port_match.py", input: "connected_port" },
      { script: "5_cable_match.py", input: "cables" },
    ];

    const resultsDir = path.join(outputDir, "Results");
    fs.mkdirSync(resultsDir, { recursive: true });

    try {
      // Run report scripts in parallel where possible to speed up processing
      const reportPromises: Array<Promise<void>> = [];
      for (const reportScript of reportScripts) {
        const segmentedInputDir = path.join(outputDir, reportScript.input);
        if (fs.existsSync(segmentedInputDir)) {
          const scriptPath = path.join(process.cwd(), "python_codes", reportScript.script);
          await storage.addJobLog(job.jobId, `Queued report script: ${reportScript.script}`);

          // start the script without awaiting here; collect promises
          const p = processPythonScript(scriptPath, segmentedInputDir, resultsDir, job.jobId)
            .then(() => storage.addJobLog(job.jobId, `${reportScript.script} completed`))
            .catch(async (err) => {
              await storage.addJobLog(job.jobId, `${reportScript.script} failed: ${err?.message || String(err)}`);
              throw err;
            });
          reportPromises.push(p);
        }
      }

      // Await all report script completions (parallel)
      if (reportPromises.length > 0) {
        await Promise.all(reportPromises);
      }

      // Run merge script to generate final PDF (after all reports finished)
      const mergeScriptPath = path.join(process.cwd(), "python_codes", "6_merge_result.py");
      await storage.addJobLog(job.jobId, "Running merge script to generate final PDF...");
      // 6_merge_result.py expects --output to be the PDF file path, not a directory
      const mergeOutputPath = path.join(resultsDir, "Merged_Result.pdf");
      const mergeResult = await processPythonScript(mergeScriptPath, resultsDir, mergeOutputPath, job.jobId);
      await storage.addJobLog(job.jobId, "Merge script completed");

      // Verify PDF was created
      const pdfPath = path.join(resultsDir, "Merged_Result.pdf");
      if (!fs.existsSync(pdfPath)) {
        throw new Error("PDF file was not created after merge script");
      }

      await storage.updateJobStatus(job.jobId, "done");
      const duration = (Date.now() - startTime) / 1000;
      await storage.addJobLog(job.jobId, `Job completed successfully with PDF in ${duration.toFixed(2)}s`);
      logger.endJobGroup(job.jobId, `completed in ${duration.toFixed(2)}s`);
    } catch (reportError: any) {
      await storage.addJobLog(job.jobId, `Report generation failed: ${reportError.message}`);
      throw new Error(`Report generation failed: ${reportError.message}`);
    }

  } catch (error: any) {
    logger.error(`Job ${job.jobId} failed: ${error?.message || ""}`, error, "JobWorker");
    await storage.updateJobStatus(job.jobId, "failed", error.message || "Unknown error");
    await storage.addJobLog(job.jobId, `Job failed: ${error.message}`);
    logger.endJobGroup(job.jobId, "failed");
  }
}

async function pollAndProcessJobs(): Promise<void> {
  if (runningJobs >= MAX_PARALLEL_JOBS) {
    return;
  }

  const availableSlots = MAX_PARALLEL_JOBS - runningJobs;
  logger.info(`pollAndProcessJobs: runningJobs=${runningJobs}, availableSlots=${availableSlots}`, "JobWorker");
  const waitingJobs = await storage.getWaitingJobs(availableSlots);
  logger.info(`pollAndProcessJobs: fetched ${waitingJobs.length} waiting job(s)`, "JobWorker");

  for (const job of waitingJobs) {
    if (runningJobs >= MAX_PARALLEL_JOBS) break;

    runningJobs++;
    logger.info(`Starting job ${job.jobId} (runningJobs now=${runningJobs})`, "JobWorker");

    processJob(job)
      .catch((err) => {
        logger.error(`Unexpected error processing job ${job.jobId}: ${err?.message || ""}`, err, "JobWorker");
      })
      .finally(() => {
        runningJobs--;
        logger.info(`Job ${job.jobId} finished (runningJobs now=${runningJobs})`, "JobWorker");
      });
  }
}

async function checkDatabaseConnection(): Promise<boolean> {
  try {
    await db.execute(sql`SELECT 1`);
    return true;
  } catch (error) {
    return false;
  }
}

export function startJobWorker(): void {
  if (isRunning) {
    logger.info("Already running", "JobWorker");
    return;
  }

  isRunning = true;
  logger.info(`Starting with ${MAX_PARALLEL_JOBS} parallel workers`, "JobWorker");

  const poll = async () => {
    if (!isRunning) return;
    
    if (!dbConnected) {
      const connected = await checkDatabaseConnection();
      if (!connected) {
        logger.info("Waiting for database connection...", "JobWorker");
        setTimeout(poll, DB_RETRY_INTERVAL_MS);
        return;
      }
      dbConnected = true;
      logger.info("Database connection established, starting job processing", "JobWorker");
    }
    
    try {
      await pollAndProcessJobs();
      setTimeout(poll, POLL_INTERVAL_MS);
    } catch (error: any) {
      const isDbError = error.code === 'ECONNREFUSED' || 
                        error.code === 'ENOTFOUND' ||
                        error.code === 'ETIMEDOUT' ||
                        error.code === 'PROTOCOL_CONNECTION_LOST' ||
                        error.message?.includes('ECONNREFUSED') ||
                        error.message?.includes('connect') ||
                        error.message?.includes('Connection');
      
      if (isDbError) {
        dbConnected = false;
        logger.info(`Database connection error, will retry in ${DB_RETRY_INTERVAL_MS / 1000} seconds...`, "JobWorker");
        setTimeout(poll, DB_RETRY_INTERVAL_MS);
      } else {
        logger.error("Poll error:", error, "JobWorker");
        setTimeout(poll, DB_RETRY_INTERVAL_MS);
      }
    }
  };

  poll();
}

export function stopJobWorker(): void {
  isRunning = false;
  logger.info("Stopped", "JobWorker");
}

export function getWorkerStatus(): { running: boolean; activeJobs: number; maxJobs: number } {
  return {
    running: isRunning,
    activeJobs: runningJobs,
    maxJobs: MAX_PARALLEL_JOBS,
  };
}

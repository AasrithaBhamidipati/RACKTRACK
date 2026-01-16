import { type User, type InsertUser, type LoginCredentials, type Upload, type InsertUpload, type Contact, type InsertContact, type UpdateProfile, type Job, type InsertJob, type JobLog, users, uploads, contacts, jobs, jobLogs, session } from "@shared/schema";
import { randomUUID } from "crypto";
import bcrypt from "bcryptjs";
import { db, pool } from "./db";
import { eq, and, desc } from "drizzle-orm";

export interface Report {
  id: string;
  userId: string;
  title: string;
  filename: string;
  pdfPath: string;
  processedImage?: string | null;
  createdAt: string;
}

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  updateUserProfile(userId: string, profile: UpdateProfile): Promise<User | undefined>;
  validateCredentials(credentials: LoginCredentials): Promise<boolean>;
  createUpload(upload: InsertUpload): Promise<Upload>;
  getAllUploads(): Promise<Upload[]>;
  getUpload(id: string): Promise<Upload | undefined>;
  createReport(report: Omit<Report, "id" | "createdAt">): Promise<Report>;
  getUserReports(userId: string): Promise<Report[]>;
  deleteReport(reportId: string): Promise<boolean>;
  getReport(reportId: string): Promise<Report | undefined>;
  createContact(contact: InsertContact): Promise<Contact>;
  getAllContacts(): Promise<Contact[]>;
  createJob(job: InsertJob): Promise<Job>;
  getJob(jobId: string): Promise<Job | undefined>;
  getJobById(id: number): Promise<Job | undefined>;
  updateJobStatus(jobId: string, status: Job["status"], errorMessage?: string): Promise<void>;
  getWaitingJobs(limit?: number): Promise<Job[]>;
  getUserJobs(userId: string): Promise<Job[]>;
  addJobLog(jobId: string, message: string): Promise<void>;
  getJobLogs(jobId: string): Promise<JobLog[]>;
}

export class MySQLStorage implements IStorage {
  private reports: Map<string, Report> = new Map();

  async getUser(id: string): Promise<User | undefined> {
    const result = await db.select().from(users).where(eq(users.id, id)).limit(1);
    return result[0];
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const result = await db.select().from(users).where(eq(users.username, username)).limit(1);
    return result[0];
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const plain = insertUser.password;
    const hashed = await bcrypt.hash(plain, 10);
    
    await db.insert(users).values({
      id,
      username: insertUser.username,
      password: hashed,
    });
    
    const user = await this.getUser(id);
    return user!;
  }

  async updateUserProfile(userId: string, profile: UpdateProfile & { profileImage?: string }): Promise<User | undefined> {
    const user = await this.getUser(userId);
    if (!user) return undefined;

    await db.update(users).set({
      email: profile.email !== undefined ? profile.email : user.email,
      fullName: profile.fullName !== undefined ? profile.fullName : user.fullName,
      profileImage: profile.profileImage !== undefined ? profile.profileImage : user.profileImage,
    }).where(eq(users.id, userId));

    return await this.getUser(userId);
  }

  async validateCredentials(credentials: LoginCredentials): Promise<boolean> {
    const user = await this.getUserByUsername(credentials.username);
    if (!user) return false;
    
    const stored = user.password || "";
    const candidate = credentials.password;

    if (stored.startsWith("$2")) {
      return await bcrypt.compare(candidate, stored);
    }

    if (stored === candidate) {
      const newHash = await bcrypt.hash(candidate, 10);
      await db.update(users).set({ password: newHash }).where(eq(users.id, user.id));
      return true;
    }

    return false;
  }

  async createUpload(insertUpload: InsertUpload): Promise<Upload> {
    const id = randomUUID();
    
    await db.insert(uploads).values({
      id,
      fileName: insertUpload.fileName,
      fileType: insertUpload.fileType,
      filePath: insertUpload.filePath,
      uploadType: insertUpload.uploadType,
    });

    const result = await db.select().from(uploads).where(eq(uploads.id, id)).limit(1);
    return result[0];
  }

  async getAllUploads(): Promise<Upload[]> {
    return await db.select().from(uploads);
  }

  async getUpload(id: string): Promise<Upload | undefined> {
    const result = await db.select().from(uploads).where(eq(uploads.id, id)).limit(1);
    return result[0];
  }

  async createReport(report: Omit<Report, "id" | "createdAt">): Promise<Report> {
    const id = randomUUID();
    const createdAt = new Date().toISOString();
    const r: Report = { ...report, id, createdAt };
    this.reports.set(id, r);
    return r;
  }

  async getUserReports(userId: string): Promise<Report[]> {
    return Array.from(this.reports.values()).filter((r) => r.userId === userId);
  }

  async deleteReport(reportId: string): Promise<boolean> {
    return this.reports.delete(reportId);
  }

  async getReport(reportId: string): Promise<Report | undefined> {
    return this.reports.get(reportId);
  }

  async createContact(insertContact: InsertContact): Promise<Contact> {
    const id = randomUUID();
    
    await db.insert(contacts).values({
      id,
      name: insertContact.name,
      email: insertContact.email,
      message: insertContact.message,
    });

    const result = await db.select().from(contacts).where(eq(contacts.id, id)).limit(1);
    return result[0];
  }

  async getAllContacts(): Promise<Contact[]> {
    return await db.select().from(contacts);
  }

  async createJob(insertJob: InsertJob): Promise<Job> {
    await db.insert(jobs).values({
      jobId: insertJob.jobId,
      userId: insertJob.userId,
      uploadId: insertJob.uploadId,
      jobType: insertJob.jobType || "cpu",
      status: insertJob.status || "waiting",
      inputPath: insertJob.inputPath,
      outputPath: insertJob.outputPath,
    });

    const result = await db.select().from(jobs).where(eq(jobs.jobId, insertJob.jobId)).limit(1);
    return result[0];
  }

  async getJob(jobId: string): Promise<Job | undefined> {
    const result = await db.select().from(jobs).where(eq(jobs.jobId, jobId)).limit(1);
    return result[0];
  }

  async getJobById(id: number): Promise<Job | undefined> {
    const result = await db.select().from(jobs).where(eq(jobs.id, id)).limit(1);
    return result[0];
  }

  async updateJobStatus(jobId: string, status: Job["status"], errorMessage?: string): Promise<void> {
    await db.update(jobs).set({
      status,
      errorMessage: errorMessage || null,
    }).where(eq(jobs.jobId, jobId));
  }

  async getWaitingJobs(limit: number = 10): Promise<Job[]> {
    return await db.select().from(jobs)
      .where(eq(jobs.status, "waiting"))
      .orderBy(jobs.createdAt)
      .limit(limit);
  }

  async getUserJobs(userId: string): Promise<Job[]> {
    return await db.select().from(jobs)
      .where(eq(jobs.userId, userId))
      .orderBy(desc(jobs.createdAt));
  }

  async addJobLog(jobId: string, message: string): Promise<void> {
    await db.insert(jobLogs).values({
      jobId,
      message,
    });
  }

  async getJobLogs(jobId: string): Promise<JobLog[]> {
    return await db.select().from(jobLogs)
      .where(eq(jobLogs.jobId, jobId))
      .orderBy(jobLogs.createdAt);
  }
}

export const storage = new MySQLStorage();

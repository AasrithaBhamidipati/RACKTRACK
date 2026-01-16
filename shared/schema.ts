import { mysqlTable, text, varchar, timestamp, int, mysqlEnum, json } from "drizzle-orm/mysql-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { sql } from "drizzle-orm";

export const session = mysqlTable("session", {
  sid: varchar("sid", { length: 255 }).primaryKey(),
  sess: json("sess").notNull(),
  expire: timestamp("expire", { fsp: 6 }).notNull(),
});

export const users = mysqlTable("users", {
  id: varchar("id", { length: 36 }).primaryKey().default(sql`(UUID())`),
  username: varchar("username", { length: 255 }).notNull().unique(),
  password: text("password").notNull(),
  email: text("email"),
  fullName: text("full_name"),
  profileImage: text("profile_image"),
  joinedAt: timestamp("joined_at").defaultNow().notNull(),
});

export const uploads = mysqlTable("uploads", {
  id: varchar("id", { length: 36 }).primaryKey().default(sql`(UUID())`),
  fileName: text("file_name").notNull(),
  fileType: text("file_type").notNull(),
  filePath: text("file_path").notNull(),
  uploadType: text("upload_type").notNull(),
  uploadedAt: timestamp("uploaded_at").defaultNow().notNull(),
});

export const contacts = mysqlTable("contacts", {
  id: varchar("id", { length: 36 }).primaryKey().default(sql`(UUID())`),
  name: text("name").notNull(),
  email: text("email").notNull(),
  message: text("message").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const jobs = mysqlTable("jobs", {
  id: int("id").primaryKey().autoincrement(),
  jobId: varchar("job_id", { length: 100 }).notNull().unique(),
  userId: varchar("user_id", { length: 36 }),
  uploadId: varchar("upload_id", { length: 36 }),
  jobType: mysqlEnum("job_type", ["cpu", "gpu"]).default("cpu"),
  status: mysqlEnum("status", ["waiting", "processing", "done", "failed"]).default("waiting"),
  inputPath: text("input_path").notNull(),
  outputPath: text("output_path"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const jobLogs = mysqlTable("job_logs", {
  id: int("id").primaryKey().autoincrement(),
  jobId: varchar("job_id", { length: 100 }).notNull(),
  message: text("message").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertUploadSchema = createInsertSchema(uploads).omit({
  id: true,
  uploadedAt: true,
});

export const insertContactSchema = createInsertSchema(contacts).omit({
  id: true,
  createdAt: true,
});

export const insertJobSchema = createInsertSchema(jobs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const loginCredentialsSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

export const updateProfileSchema = z.object({
  email: z.string().email("Invalid email address").optional(),
  fullName: z.string().min(1, "Name is required").optional(),
  profileImage: z.string().optional(),
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type LoginCredentials = z.infer<typeof loginCredentialsSchema>;
export type UpdateProfile = z.infer<typeof updateProfileSchema>;
export type InsertUpload = z.infer<typeof insertUploadSchema>;
export type Upload = typeof uploads.$inferSelect;
export type InsertContact = z.infer<typeof insertContactSchema>;
export type Contact = typeof contacts.$inferSelect;
export type InsertJob = z.infer<typeof insertJobSchema>;
export type Job = typeof jobs.$inferSelect;
export type JobLog = typeof jobLogs.$inferSelect;
export type Session = typeof session.$inferSelect;

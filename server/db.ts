import { drizzle } from "drizzle-orm/mysql2";
import mysql from "mysql2/promise";
import * as schema from "@shared/schema";

const DATABASE_URL = process.env.MYSQL_URL || "mysql://root:admin123@localhost:3306/RACKTRACK";

const pool = mysql.createPool(DATABASE_URL);

export const db = drizzle(pool, { schema, mode: "default" });
export { pool };

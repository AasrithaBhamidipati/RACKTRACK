function timeStamp(): string {
  return new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

export function info(message: string, source = "server") {
  console.log(`${timeStamp()} [${source}] ${message}`);
}

export function error(message: string, err?: unknown, source = "server") {
  if (err) {
    console.error(`${timeStamp()} [${source}] ${message}`, err);
  } else {
    console.error(`${timeStamp()} [${source}] ${message}`);
  }
}

export function startJobGroup(jobId: string, title?: string) {
  const header = `=== Job ${jobId} ${title ? `- ${title}` : "started"} ===`;
  console.log(`${timeStamp()} [job] ${header}`);
}

export function endJobGroup(jobId: string, summary?: string) {
  const footer = `=== Job ${jobId} finished ${summary ? `- ${summary}` : ""} ===`;
  console.log(`${timeStamp()} [job] ${footer}`);
}

export function streamToJob(jobId: string) {
  return (data: string | Buffer) => {
    const text = data.toString().trim();
    if (!text) return;
    const prefix = `${timeStamp()} [job:${jobId}]`;
    // Print each non-empty line separately to keep output clean
    text.split(/\r?\n/).forEach((line) => {
      if (line.trim()) console.log(prefix, line.trim());
    });
  };
}

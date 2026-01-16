import { useEffect, useRef } from "react";

interface CanvasVideoProps {
  src: string;
  className?: string;
  poster?: string;
  playbackRate?: number;
}

const CanvasVideo = ({ src, className = "", poster, playbackRate = 1 }: CanvasVideoProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const video = document.createElement("video");
    video.src = src;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.autoplay = true;
    video.style.display = "none";
    videoRef.current = video;

    let animationId: number;

    const drawFrame = () => {
      if (video.readyState >= 2) {
        canvas.width = video.videoWidth || canvas.offsetWidth;
        canvas.height = video.videoHeight || canvas.offsetHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      animationId = requestAnimationFrame(drawFrame);
    };

    video.addEventListener("loadeddata", () => {
      video.playbackRate = playbackRate;
      video.play().catch(() => {});
      drawFrame();
    });

    video.load();

    return () => {
      cancelAnimationFrame(animationId);
      video.pause();
      video.src = "";
      videoRef.current = null;
    };
  }, [src]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ 
        backgroundImage: poster ? `url(${poster})` : undefined,
        backgroundSize: "cover",
        backgroundPosition: "center"
      }}
    />
  );
};

export default CanvasVideo;

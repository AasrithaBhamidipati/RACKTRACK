import { useState, useCallback, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, X, Video, FileImage } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface FilePreview {
  file: File;
  preview: string;
}

interface FileDropZoneProps {
  onFilesSelected: (files: File[]) => void;
  acceptedTypes: string;
  maxFiles: number;
  uploadType: string;
  accentColor: string;
}

export function FileDropZone({
  onFilesSelected,
  acceptedTypes,
  maxFiles,
  uploadType,
  accentColor,
}: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FilePreview[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    onFilesSelected(selectedFiles.map((fp) => fp.file));
  }, [selectedFiles, onFilesSelected]);

  useEffect(() => {
    return () => {
      selectedFiles.forEach((fp) => URL.revokeObjectURL(fp.preview));
    };
  }, [selectedFiles]);

  const createPreview = useCallback(async (file: File): Promise<string> => {
    if (file.type.startsWith("image/") || file.type.startsWith("video/")) {
      return URL.createObjectURL(file);
    }
    return "";
  }, []);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files) return;

      const fileArray = Array.from(files);
      const validFiles = fileArray.filter((file) => {
        const typeMatch = acceptedTypes.split(",").some((type) => {
          const trimmedType = type.trim();
          if (trimmedType.startsWith(".")) {
            return file.name.toLowerCase().endsWith(trimmedType.toLowerCase());
          }
          return file.type.match(new RegExp(trimmedType.replace("*", ".*")));
        });
        return typeMatch;
      });

      if (selectedFiles.length + validFiles.length > maxFiles) {
        toast({
          title: "File limit exceeded",
          description: `You can only upload up to ${maxFiles} file(s)`,
          variant: "destructive",
        });
        return;
      }

      const newPreviews = await Promise.all(
        validFiles.map(async (file) => ({
          file,
          preview: await createPreview(file),
        }))
      );

      setSelectedFiles((prev) => [...prev, ...newPreviews]);
    },
    [acceptedTypes, maxFiles, selectedFiles.length, createPreview, toast]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleBrowse = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files);
    },
    [handleFiles]
  );

  const removeFile = useCallback((index: number) => {
    setSelectedFiles((prev) => {
      const newFiles = [...prev];
      URL.revokeObjectURL(newFiles[index].preview);
      newFiles.splice(index, 1);
      return newFiles;
    });
  }, []);

  if (selectedFiles.length === 0) {
    return (
      <div
        className={`relative border-2 border-dashed rounded-xl transition-all duration-300 ${
          isDragging
            ? "border-white/40 bg-white/5"
            : "border-white/20 glass hover:border-white/30"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="dropzone-empty"
      >
        <div className="p-12 flex flex-col items-center justify-center gap-5 text-center">
          <div className="relative">
            <div className={`absolute inset-0 bg-gradient-to-br ${accentColor} opacity-30 rounded-xl blur-xl`} />
            <div className={`relative w-16 h-16 rounded-xl bg-gradient-to-br ${accentColor} flex items-center justify-center shadow-lg`}>
              <Upload className="w-8 h-8 text-white" strokeWidth={2} />
            </div>
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-white mb-2">
              Drop your files here
            </h3>
            <p className="text-white/60 text-sm mb-3">
              or click the button below to browse
            </p>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-strong border border-white/20">
              <FileImage className="w-3.5 h-3.5 text-white/70" />
              <p className="text-xs text-white/70 font-medium">
                Supports {acceptedTypes.split(",").map(f => f.split("/")[1]?.toUpperCase() || f).join(", ")}
              </p>
            </div>
          </div>
          
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleBrowse}
            accept={acceptedTypes}
            multiple={maxFiles > 1}
            data-testid="input-file"
          />
          
          <label htmlFor="file-upload" className="cursor-pointer">
            <Button
              type="button"
              className={`bg-gradient-to-r ${accentColor} hover:opacity-90 text-white font-semibold px-8 shadow-lg`}
              onClick={() => document.getElementById("file-upload")?.click()}
              data-testid="button-browse"
            >
              <Upload className="w-4 h-4 mr-2" />
              Browse Files
            </Button>
          </label>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="dropzone-preview">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {selectedFiles.map((filePreview, index) => (
          <Card
            key={index}
            className="group relative overflow-hidden glass-strong border-2 border-white/10 hover:border-white/30 transition-all duration-300"
            data-testid={`card-file-${index}`}
          >
            <div className="aspect-square relative bg-slate-950/50">
              {filePreview.file.type.startsWith("image/") ? (
                <img
                  src={filePreview.preview}
                  alt={filePreview.file.name}
                  className="w-full h-full object-cover"
                  data-testid={`img-preview-${index}`}
                />
              ) : filePreview.file.type.startsWith("video/") ? (
                <div className="relative w-full h-full">
                  <video
                    src={filePreview.preview}
                    className="w-full h-full object-cover"
                    data-testid={`video-preview-${index}`}
                  />
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${accentColor} flex items-center justify-center`}>
                      <Video className="w-6 h-6 text-white" strokeWidth={2} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Upload className="w-10 h-10 text-white/40" />
                </div>
              )}
              
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  type="button"
                  variant="destructive"
                  size="icon"
                  className="w-8 h-8 rounded-lg"
                  onClick={() => removeFile(index)}
                  data-testid={`button-remove-${index}`}
                  aria-label="Remove file"
                >
                  <X className="w-4 h-4" strokeWidth={2.5} />
                </Button>
              </div>
            </div>
            
            <div className="p-3 bg-gradient-to-b from-white/5 to-transparent">
              <p
                className="text-xs text-white truncate font-semibold mb-1.5"
                title={filePreview.file.name}
                data-testid={`text-filename-${index}`}
              >
                {filePreview.file.name}
              </p>
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="px-2 py-0.5 rounded glass-strong text-white/70" data-testid={`text-filetype-${index}`}>
                  {filePreview.file.type.split("/")[1]?.toUpperCase() || "File"}
                </span>
                <span className="text-white/50">
                  {(filePreview.file.size / 1024).toFixed(1)} KB
                </span>
              </div>
            </div>
            
            <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${accentColor} opacity-0 group-hover:opacity-100 transition-opacity`} />
          </Card>
        ))}
      </div>
      
      {selectedFiles.length < maxFiles && (
        <div>
          <input
            type="file"
            id="file-upload-more"
            className="hidden"
            onChange={handleBrowse}
            accept={acceptedTypes}
            multiple={maxFiles > 1}
            data-testid="input-file-more"
          />
          <label htmlFor="file-upload-more" className="cursor-pointer">
            <Button
              type="button"
              variant="outline"
              className="w-full border-2 border-white/20 glass text-white hover:bg-white/10 hover:border-white/40 text-sm"
              onClick={() =>
                document.getElementById("file-upload-more")?.click()
              }
              data-testid="button-add-more"
            >
              <Upload className="w-4 h-4 mr-2" />
              Add More ({selectedFiles.length}/{maxFiles})
            </Button>
          </label>
        </div>
      )}
    </div>
  );
}

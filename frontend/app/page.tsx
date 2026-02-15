"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export default function Home() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const audioUrl = useMemo(() => {
    if (!audioBlob) return null;
    return URL.createObjectURL(audioBlob);
  }, [audioBlob]);

  const cleanupStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  useEffect(() => {
    return () => {
      cleanupStream();
    };
  }, []);

  const startRecording = async () => {
    setError(null);
    setAudioBlob(null);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        cleanupStream();
      };

      recorder.start();
      setIsRecording(true);
    } catch {
      setError("Microphone access failed. Please allow permission and try again.");
      cleanupStream();
    }
  };

  const stopRecording = () => {
    if (!recorderRef.current || recorderRef.current.state !== "recording") return;
    recorderRef.current.stop();
    setIsRecording(false);
  };

  const handleRecordClick = async () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    await startRecording();
  };

  const handlePrepareForApi = () => {
    if (!audioBlob) return;

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    // Replace this with your upload call:
    // await fetch("/api/your-endpoint", { method: "POST", body: formData });
    console.log("Audio is ready for API request", formData.get("audio"));
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-6 text-center bg-amber-50">
      <h1 className="text-3xl font-semibold">Audio Recorder</h1>

      <button
        type="button"
        onClick={handleRecordClick}
        className="rounded-full bg-black px-6 py-3 text-white transition hover:opacity-85"
      >
        {isRecording ? "Stop Recording" : "Start Recording"}
      </button>

      <p className="text-sm text-zinc-600">
        {isRecording
          ? "Recording in progress..."
          : audioBlob
            ? "Recording stopped. Clip is ready."
            : "Click start to begin recording."}
      </p>

      {error ? <p className="text-sm text-red-600">{error}</p> : null}

      {audioUrl ? (
        <div className="flex flex-col items-center gap-3">
          <audio controls src={audioUrl} />
          <button
            type="button"
            onClick={handlePrepareForApi}
            className="rounded-full border border-black px-5 py-2 transition hover:bg-black hover:text-white"
          >
            Prepare API Payload
          </button>
        </div>
      ) : null}
    </main>
  );
}

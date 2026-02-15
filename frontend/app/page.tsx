"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ApiResult = Record<string, unknown>;

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function floatTo16BitPCM(view: DataView, offset: number, input: Float32Array): void {
  let cursor = offset;
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(cursor, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    cursor += 2;
  }
}

function encodeWav(audioBuffer: AudioBuffer): Blob {
  const numberOfChannels = 1;
  const sampleRate = audioBuffer.sampleRate;
  const bitsPerSample = 16;
  const blockAlign = (numberOfChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataLength = audioBuffer.length * blockAlign;

  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  let offset = 0;
  const writeString = (value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset, value.charCodeAt(i));
      offset += 1;
    }
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + dataLength, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, numberOfChannels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, byteRate, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bitsPerSample, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, dataLength, true);
  offset += 4;

  const mono = new Float32Array(audioBuffer.length);
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let i = 0; i < audioBuffer.length; i += 1) {
      mono[i] += channelData[i] / audioBuffer.numberOfChannels;
    }
  }
  floatTo16BitPCM(view, offset, mono);

  return new Blob([view], { type: "audio/wav" });
}

async function convertBlobToWav(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return encodeWav(audioBuffer);
  } finally {
    await audioContext.close();
  }
}

async function postAudio(endpoint: string, audioBlob: Blob, filename: string): Promise<ApiResult> {
  const formData = new FormData();
  formData.append("audio", audioBlob, filename);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  const payload = (await response.json()) as ApiResult;
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload);
    throw new Error(detail);
  }

  return payload;
}

export default function Home() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [transcriptionResult, setTranscriptionResult] = useState<ApiResult | null>(null);
  const [audioProcessResult, setAudioProcessResult] = useState<ApiResult | null>(null);

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
    setTranscriptionResult(null);
    setAudioProcessResult(null);
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
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
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

  const prepareUploadBlob = async (): Promise<Blob> => {
    if (!audioBlob) {
      throw new Error("No recording available");
    }

    try {
      return await convertBlobToWav(audioBlob);
    } catch {
      return audioBlob;
    }
  };

  const runTranscription = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const uploadBlob = await prepareUploadBlob();
      const result = await postAudio("/api/transcription", uploadBlob, "recording.wav");
      setTranscriptionResult(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Transcription request failed";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const runAudioProcess = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const uploadBlob = await prepareUploadBlob();
      const result = await postAudio("/api/audio-process", uploadBlob, "recording.wav");
      setAudioProcessResult(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Audio processing request failed";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const runBoth = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const uploadBlob = await prepareUploadBlob();
      const [transcription, audioProcess] = await Promise.allSettled([
        postAudio("/api/transcription", uploadBlob, "recording.wav"),
        postAudio("/api/audio-process", uploadBlob, "recording.wav"),
      ]);

      if (transcription.status === "fulfilled") {
        setTranscriptionResult(transcription.value);
      } else {
        setError(`Transcription failed: ${transcription.reason instanceof Error ? transcription.reason.message : "Unknown error"}`);
      }

      if (audioProcess.status === "fulfilled") {
        setAudioProcessResult(audioProcess.value);
      } else {
        const processError = `Audio process failed: ${audioProcess.reason instanceof Error ? audioProcess.reason.message : "Unknown error"}`;
        setError((current) => (current ? `${current} | ${processError}` : processError));
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-amber-50 px-6 py-10 text-zinc-900">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
        <h1 className="text-3xl font-semibold">MinuteZero Audio Console</h1>

        <div className="rounded-2xl border border-zinc-300 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleRecordClick}
              className="rounded-full bg-black px-6 py-3 text-white transition hover:opacity-85 disabled:opacity-50"
              disabled={isSubmitting}
            >
              {isRecording ? "Stop Recording" : "Start Recording"}
            </button>

            <button
              type="button"
              onClick={runBoth}
              disabled={!audioBlob || isSubmitting || isRecording}
              className="rounded-full border border-black px-5 py-2 transition hover:bg-black hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              Send to Both APIs
            </button>

            <button
              type="button"
              onClick={runTranscription}
              disabled={!audioBlob || isSubmitting || isRecording}
              className="rounded-full border border-zinc-400 px-5 py-2 transition hover:border-black disabled:cursor-not-allowed disabled:opacity-40"
            >
              Transcription Only
            </button>

            <button
              type="button"
              onClick={runAudioProcess}
              disabled={!audioBlob || isSubmitting || isRecording}
              className="rounded-full border border-zinc-400 px-5 py-2 transition hover:border-black disabled:cursor-not-allowed disabled:opacity-40"
            >
              Audio Process Only
            </button>
          </div>

          <p className="mt-4 text-sm text-zinc-600">
            {isRecording
              ? "Recording in progress..."
              : isSubmitting
                ? "Sending audio to API..."
                : audioBlob
                  ? "Recording stopped. Clip is ready to submit."
                  : "Click start to begin recording."}
          </p>

          {audioUrl ? (
            <div className="mt-4">
              <audio controls src={audioUrl} className="w-full" />
            </div>
          ) : null}

          {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <section className="rounded-2xl border border-zinc-300 bg-white p-4 shadow-sm">
            <h2 className="mb-2 text-lg font-semibold">Transcription Result</h2>
            <pre className="max-h-80 overflow-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
              {transcriptionResult ? JSON.stringify(transcriptionResult, null, 2) : "No transcription result yet."}
            </pre>
          </section>

          <section className="rounded-2xl border border-zinc-300 bg-white p-4 shadow-sm">
            <h2 className="mb-2 text-lg font-semibold">Audio Process Result</h2>
            <pre className="max-h-80 overflow-auto rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
              {audioProcessResult ? JSON.stringify(audioProcessResult, null, 2) : "No audio process result yet."}
            </pre>
          </section>
        </div>

        <p className="text-xs text-zinc-500">
          Backend URL: <code>{API_BASE_URL}</code>
        </p>
      </div>
    </main>
  );
}

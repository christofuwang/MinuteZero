"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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

async function postAudioAnalyze(audioBlob: Blob, filename: string) {
  const formData = new FormData();
  formData.append("audio", audioBlob, filename);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(async () => {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  });

  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string" ? payload.detail : JSON.stringify(payload ?? {});
    throw new Error(detail);
  }

  return payload as {
    transcript?: string;
    emotion_dims?: Record<string, number>;
    acoustic_scores?: Record<string, number>;
    recommendation?: string;
  };
}

function formatMetrics(
  emotionDims?: Record<string, number>,
  acousticScores?: Record<string, number>
): string {
  const lines: string[] = [];

  // Emotion
  lines.push("Emotion Dimensions (approx 0..1):");
  if (emotionDims && Object.keys(emotionDims).length > 0) {
    const order = ["arousal", "dominance", "valence"];
    for (const key of order) {
      const v = emotionDims[key];
      if (typeof v === "number") lines.push(`  ${key}: ${v.toFixed(4)}`);
    }
    // include any extras
    for (const [k, v] of Object.entries(emotionDims)) {
      if (order.includes(k)) continue;
      if (typeof v === "number") lines.push(`  ${k}: ${v.toFixed(4)}`);
    }
  } else {
    lines.push("  (no emotion dims returned)");
  }

  lines.push("");
  lines.push("Acoustic signals:");
  if (acousticScores && Object.keys(acousticScores).length > 0) {
    const rms = acousticScores.rms;
    const zcr = acousticScores.zcr;
    const yell = acousticScores.yell_score;
    const whisper = acousticScores.whisper_score;
    const concern = acousticScores.dispatch_concern;

    if (typeof rms === "number" && typeof zcr === "number") {
      lines.push(`  rms: ${rms.toFixed(5)} | zcr: ${zcr.toFixed(3)}`);
    }
    if (typeof yell === "number" && typeof whisper === "number") {
      lines.push(`  yell_score: ${yell.toFixed(3)} | whisper_score: ${whisper.toFixed(3)}`);
    }
    if (typeof concern === "number") {
      lines.push(`  dispatch_concern: ${concern.toFixed(4)}`);
    }

    // include any extras
    const known = new Set(["rms", "zcr", "yell_score", "whisper_score", "dispatch_concern"]);
    for (const [k, v] of Object.entries(acousticScores)) {
      if (known.has(k)) continue;
      if (typeof v === "number") lines.push(`  ${k}: ${v}`);
      else lines.push(`  ${k}: ${String(v)}`);
    }
  } else {
    lines.push("  (no acoustic scores returned)");
  }

  return lines.join("\n");
}

export default function Home() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ✅ Text-only outputs for the 3 boxes
  const [transcriptText, setTranscriptText] = useState<string>("");
  const [metricsText, setMetricsText] = useState<string>("");
  const [summaryText, setSummaryText] = useState<string>("");

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
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  useEffect(() => {
    return () => cleanupStream();
  }, []);

  const startRecording = async () => {
    setError(null);
    setAudioBlob(null);

    // Clear boxes
    setTranscriptText("");
    setMetricsText("");
    setSummaryText("");

    chunksRef.current = [];

    try {
      //const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }});
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
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
    if (!audioBlob) throw new Error("No recording available");
    try {
      return await convertBlobToWav(audioBlob);
    } catch {
      return audioBlob;
    }
  };

  const runAnalyze = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const uploadBlob = await prepareUploadBlob();
      const result = await postAudioAnalyze(uploadBlob, "recording.wav");

      // ✅ Route each piece to its correct box
      setTranscriptText(result.transcript ? result.transcript : "");
      setMetricsText(formatMetrics(result.emotion_dims, result.acoustic_scores));
      setSummaryText(result.recommendation ? result.recommendation : "");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analyze request failed";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-br from-[#2b2d42] to-[#937d92] px-6 py-10 text-white">
      {/* Ambient glow blobs */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -left-24 -top-24 h-[420px] w-[420px] rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -right-24 top-1/3 h-[520px] w-[520px] rounded-full bg-[#937d92]/25 blur-3xl" />
        <div className="absolute left-1/3 -bottom-28 h-[520px] w-[520px] rounded-full bg-[#2b2d42]/35 blur-3xl" />
      </div>

      {/* Big faint background logo */}
      <div className="pointer-events-none absolute inset-0 z-0 flex items-center justify-center">
        <img
          src="/LogoFull_Final.png"
          alt=""
          className="w-[160vw] max-w-none opacity-[0.12] blur-xl"
        />
      </div>

      <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
        {/* Header with logo + pulse ring when recording */}
        <div className="flex items-center gap-4">
          <div className="relative">
            {/* pulse ring */}
            {isRecording ? (
              <span className="absolute inset-0 -z-10 rounded-full bg-white/20 blur-md animate-pulse" />
            ) : null}

            <div className="grid h-12 w-12 place-items-center rounded-full bg-white/10 backdrop-blur border border-white/20 shadow-lg">
              <img src="/LogoFull_Final.png" alt="MinuteZero" className="h-auto w-9" />
            </div>
          </div>

          <div>
            <h1 className="text-3xl font-semibold tracking-tight">MinuteZero</h1>
            <p className="text-sm text-white/70">
              Audio Console • Transcript • Metrics • Dispatch Summary
            </p>
          </div>
        </div>

        {/* Recorder card (glass) */}
        <div className="relative rounded-2xl border border-white/20 bg-white/10 p-5 shadow-xl backdrop-blur-xl">
          {/* watermark logo in card */}
          <img
            src="/LogoFull_Final.png"
            alt=""
            className="pointer-events-none absolute bottom-4 right-4 w-24 opacity-[0.08]"
          />

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleRecordClick}
              className="rounded-full bg-white/15 px-6 py-3 text-white backdrop-blur border border-white/20 shadow-lg hover:bg-white/20 transition disabled:opacity-50"
              disabled={isSubmitting}
            >
              {isRecording ? "Stop Recording" : "Start Recording"}
            </button>

            <button
              type="button"
              onClick={runAnalyze}
              disabled={!audioBlob || isSubmitting || isRecording}
              className="rounded-full bg-black/30 px-5 py-2 text-white border border-white/20 shadow hover:bg-black/40 transition disabled:cursor-not-allowed disabled:opacity-40"
            >
              Analyze (Transcript + Metrics + Summary)
            </button>
          </div>

          <p className="mt-4 text-sm text-white/70">
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

          {error ? <p className="mt-3 text-sm text-red-200">{error}</p> : null}
        </div>

        {/* Panels */}
        <div className="grid gap-4 md:grid-cols-2">
          <section className="rounded-2xl border border-white/20 bg-white/10 p-4 shadow-xl backdrop-blur-xl">
            <h2 className="mb-2 text-lg font-semibold">Transcription</h2>
            <div className="max-h-80 overflow-auto rounded-md bg-black/30 p-3 text-sm text-white/90 whitespace-pre-wrap border border-white/10 font-mono leading-relaxed">
              {transcriptText ? transcriptText : "No transcript yet."}
            </div>
          </section>

          <section className="rounded-2xl border border-white/20 bg-white/10 p-4 shadow-xl backdrop-blur-xl">
            <h2 className="mb-2 text-lg font-semibold">Audio Process (Emotion + Acoustic)</h2>
            <div className="max-h-80 overflow-auto rounded-md bg-black/30 p-3 text-sm text-white/90 whitespace-pre-wrap border border-white/10 font-mono leading-relaxed">
              {metricsText ? metricsText : "No audio result yet."}
            </div>
          </section>
        </div>

        <section className="rounded-2xl border border-white/20 bg-white/10 p-4 shadow-xl backdrop-blur-xl">
          <h2 className="mb-2 text-lg font-semibold">Dispatch Summary (Agent Output)</h2>
          <div className="max-h-96 overflow-auto rounded-md bg-black/30 p-3 text-sm text-white/90 whitespace-pre-wrap border border-white/10 font-mono leading-relaxed">
            {summaryText ? summaryText : "No summary yet."}
          </div>
        </section>
      </div>
    </main>
  );
}


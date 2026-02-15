import speech_recognition as sr
import time

def main():
    r = sr.Recognizer()
    r.energy_threshold = 300  # you can tweak this
    r.dynamic_energy_threshold = True

    mic = sr.Microphone()  # default system mic

    print("Calibrating for ambient noise... (stay quiet for 1 second)")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)

    print("Listening! Speak into your mic. Press Ctrl+C to stop.\n")

    def callback(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)  # sends audio to Google
            if text.strip():
                print(text)
        except sr.UnknownValueError:
            # Couldn’t understand that chunk; ignore quietly
            pass
        except sr.RequestError as e:
            print(f"[API error] {e}")

    stop_listening = r.listen_in_background(mic, callback, phrase_time_limit=3)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()

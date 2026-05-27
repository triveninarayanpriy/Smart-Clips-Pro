# 🎬 Smart Clips Pro: Podcast-to-Reel Automation (v1.0)

A professional, local-first video processing pipeline that converts long-form podcasts into highly engaging vertical (9:16) shorts. 

Unlike basic AI clipping tools that make jarring mid-sentence cuts and charge monthly API fees, **Smart Clips Pro** utilizes hand-curated narrative loops, sub-line timestamp interpolation, and single-pass FFmpeg rendering to create perfectly synced, "karaoke-style" Shorts running 100% locally on your machine.

---

## ✨ Core Features
* **Karaoke-Style Subtitles:** Whisper transcript segments are automatically chunked into ≤6-word sub-lines. Each line receives a proportional timestamp slice, ensuring text never overflows the vertical frame and perfectly matches the speaker's cadence.
* **Flicker-Free Rendering:** Engineered with 50ms subtitle end-caps and whitespace stripping to eliminate empty subtitle blocks and frame flickering.
* **Perfect Video Cuts:** Video clipping is aligned strictly to validated sentence boundaries to prevent mid-breath or mid-word audio chops.
* **Single-Pass FFmpeg Encoding:** Optimized CLI commands perform fast-seeking, center-cropping, subtitle burn-in, and H.264 encoding in a single pass, drastically reducing generation loss and render time.
* **Zero API Costs:** Runs entirely locally. No external LLM tokens or cloud rendering fees required.

---

## 🛠️ Prerequisites
Before running this pipeline, ensure your system has the following installed and added to your system `PATH`:

1. **Python 3.7+**
2. **[FFmpeg](https://ffmpeg.org/download.html)** (Required for video editing and subtitle rendering)
3. **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** (Required for downloading source video)
4. **[OpenAI Whisper CLI](https://github.com/openai/whisper)** (Required for local transcription)
   * *Install via:* `pip install -U openai-whisper`

---

## 📁 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone (https://github.com/triveninarayanpriy/Smart-Clips-Pro.git)
   cd Smart-Clips-Pro

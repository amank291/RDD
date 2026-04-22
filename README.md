# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using facial landmark analysis. Detects eye closure (EAR) and head nodding (pitch angle) via webcam, triggers audible beeps, and sends WhatsApp alerts via Twilio when drowsiness is detected.

---

## Features

- Real-time Eye Aspect Ratio (EAR) monitoring via MediaPipe FaceMesh
- Head-pose estimation using `solvePnP` (pitch / yaw / roll)
- Audible multi-beep alert on drowsiness or nodding
- WhatsApp alert via Twilio with EAR value and timestamp
- Live EAR graph with scrolling history
- On-screen HUD showing EAR, pitch, alert count, and drowsy percentage
- Adjustable threshold and frame-count via keyboard at runtime

---

## Requirements

- Python 3.8+
- Windows (uses `winsound` for audio beeps)
- Webcam

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/drowsiness-detector.git
cd drowsiness-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe scipy numpy twilio
```

---

## Twilio WhatsApp Setup

The system sends WhatsApp alerts using the [Twilio API](https://www.twilio.com/).

1. Sign up at [twilio.com](https://www.twilio.com) and get your **Account SID** and **Auth Token**
2. Join the Twilio WhatsApp Sandbox by sending `join <sandbox-keyword>` to `+1 415 523 8886` from your WhatsApp
3. Open `main.py` and update the credentials at the top of the file:

```python
TWILIO_SID   = "YOUR_ACCOUNT_SID"
TWILIO_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM  = "+14155238886"       # Twilio sandbox number (default)
ALERT_TO     = "+91XXXXXXXXXX"      # Your WhatsApp number with country code
```

> **Note:** Keep your credentials private. Do not commit them to version control. Use environment variables in production.

---

## Usage

```bash
python main.py
```

Two windows will open:
- **Drowsiness Detector** — live webcam feed with HUD overlay
- **EAR Live Graph** — scrolling EAR history with threshold line

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit and print session summary |
| `+` / `=` | Increase EAR threshold by 0.01 |
| `-` | Decrease EAR threshold by 0.01 |
| `]` | Increase consecutive frame count by 1 |
| `[` | Decrease consecutive frame count by 1 |
| `T` | Send a test WhatsApp alert |

---

## Configuration

All key parameters are defined at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | `0.20` | EAR value below which eye is considered closed |
| `CONSEC_FRAMES` | `15` | Consecutive frames below threshold to trigger alert |
| `NOD_THRESHOLD` | `20` | Pitch angle (degrees) to detect head nodding |
| `NOD_FRAMES` | `25` | Consecutive nod frames to trigger alert |
| `WHATSAPP_COOLDOWN` | `60` | Seconds between WhatsApp alerts |
| `BEEP_COOLDOWN` | `3` | Seconds between audio beeps |

---

## How It Works

```
Webcam → MediaPipe FaceMesh → EAR computation → Threshold check → Alert
                            ↘ Head pose (pitch) → Nod detection → Alert
```

1. **EAR (Eye Aspect Ratio)** is computed from 6 eye landmarks per eye using the formula:

   ```
   EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
   ```

2. If `EAR < EAR_THRESHOLD` for `CONSEC_FRAMES` consecutive frames → drowsiness alert
3. If head `pitch > NOD_THRESHOLD` for `NOD_FRAMES` consecutive frames → nodding alert
4. Both triggers fire an audible beep and (after cooldown) a WhatsApp message

---

## Project Structure

```
drowsiness-detector/
│
├── main.py          # Main application script
└── README.md        # This file
```

---

## Session Summary

On exit, the console prints a summary:

```
══ SESSION SUMMARY ══════════════════
  Total frames  : 3420
  Drowsy frames : 148  (4.3%)
  Alerts fired  : 2
═════════════════════════════════════
```

---

## Limitations

- `winsound` is Windows-only; replace with `playsound` or `pygame` for cross-platform use
- Requires good lighting and a front-facing camera for reliable landmark detection
- Twilio WhatsApp sandbox requires the recipient to opt-in first

---

## License

MIT License. Feel free to use and modify for research or personal projects.

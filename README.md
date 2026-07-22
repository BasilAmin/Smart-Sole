# Smart Sole

A lightweight, modular foot-sensing research prototype for real-time pressure monitoring, gait and behaviour classification, and anomaly detection.

[Project page](https://basilamin.com/projects/smart-sole/) ·
[Project book](./Smart%20Sole%20Project%20book.pdf) ·
[Presentation poster](./Smart%20Sole%20Presentation%20Poster.pdf) ·
[MIT License](./LICENSE)

## Overview

Smart Sole explores how embedded pressure sensors and machine learning could help identify unusual foot-pressure patterns associated with diabetic peripheral neuropathy.

The prototype places force-sensitive resistors at three areas of the foot:

- Heel
- Lateral arch
- Medial metatarsal

A Seeed Studio XIAO ESP32-S3 collects the sensor readings and transmits them over Bluetooth Low Energy. A Python application receives the data and experiments with a hybrid analysis pipeline combining neural-network behaviour classification with Isolation Forest anomaly detection.

> [!IMPORTANT]
> Smart Sole Version 1 is an experimental research prototype. It is not a certified medical device and must not be used for diagnosis, treatment, safety-critical monitoring, or as a substitute for professional medical care.

## Motivation

Diabetic peripheral neuropathy can reduce sensation in the feet, making potentially harmful pressure patterns more difficult to notice.

Smart Sole was inspired by my grandmother’s experience with the condition. The project investigates whether a wearable sensing system could act as a virtual feedback loop: observe pressure and movement, detect unusual patterns, and eventually provide useful feedback to the wearer.

## System Architecture

```text
┌──────────────────────────────┐
│ Force-Sensitive Resistors    │
│                              │
│  • Heel                      │
│  • Lateral arch              │
│  • Medial metatarsal         │
└──────────────┬───────────────┘
               │ Analogue readings
               ▼
┌──────────────────────────────┐
│ Seeed Studio XIAO ESP32-S3   │
│                              │
│  • Sensor sampling           │
│  • BLE GATT server           │
│  • Data formatting           │
└──────────────┬───────────────┘
               │ BLE notifications
               ▼
┌──────────────────────────────┐
│ Python Analysis Pipeline     │
│                              │
│  • Bleak BLE client          │
│  • Data preprocessing        │
│  • Behaviour classification │
│  • Anomaly detection         │
│  • Live visualisation        │
└──────────────────────────────┘
```

## Features

- Three-zone plantar-pressure sensing
- Wireless BLE data transmission
- Live sensor-data visualisation
- Experimental classification of standing, sitting, walking, limping, and pressure-avoidance behaviours
- Isolation Forest-based anomaly detection
- Pre-trained Keras models and sample datasets
- KiCad electronics files
- 3D-printable sole model
- PlatformIO-based firmware
- Project book and presentation poster

## Hardware

The Version 1 firmware is configured for the following components:

| Component | Purpose |
| --- | --- |
| Seeed Studio XIAO ESP32-S3 | Sensor acquisition and BLE communication |
| Heel FSR | Measures pressure near the heel |
| Lateral-arch FSR | Measures pressure near the outer arch |
| Medial-metatarsal FSR | Measures pressure near the forefoot |
| Custom sole and electronics | Holds and connects the system components |

### Firmware Pin Mapping

| Sensor | ESP32-S3 Pin |
| --- | --- |
| Heel | `A0` |
| Lateral arch | `A1` |
| Medial metatarsal | `A2` |

The current firmware transmits raw analogue readings. These values have not yet been converted into calibrated force or pressure units.

## BLE Data Format

The firmware publishes sensor readings as a UTF-8 string:

```text
HEEL <value>, LATERAL ARCH <value>, MEDIAL METATARSAL <value>, TIME <seconds>
```

Example:

```text
HEEL 1270, LATERAL ARCH 842, MEDIAL METATARSAL 2315, TIME 42
```

The firmware and Python clients use a custom BLE service and characteristic defined in the source code.

## Repository Structure

```text
Smart-Sole/
├── CAD/
│   ├── Schematic/                 # Electronics schematic files
│   ├── Smart_Sole/                # KiCad project files
│   ├── FSR LIB.kicad_sym          # FSR symbol library
│   └── SSFOOT.stl                 # 3D-printable sole model
│
├── Data management/
│   ├── Behaviour_regocnition_BLE.py
│   └── Smart_Sole_Models/         # Model-training experiments
│
├── Smart Sole Software/
│   ├── Behavioural detection model/
│   │   ├── Behaviour_model_training.py
│   │   └── *.keras                # Trained classification models
│   ├── Smart_Sole_Models/
│   │   ├── Smart_Sole_System_main.py
│   │   ├── Main.py
│   │   ├── Iso_forest_test.py
│   │   └── *.csv                  # Training and reference datasets
│   └── BLE_ScanForMAC             # BLE device discovery utility
│
├── SmartSole Firmware/
│   ├── src/main.cpp               # ESP32-S3 firmware
│   └── platformio.ini             # PlatformIO configuration
│
├── Smart Sole Presentation Poster.pdf
├── Smart Sole Project book.pdf
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

For the firmware:

- Seeed Studio XIAO ESP32-S3
- Three force-sensitive resistors
- A computer with PlatformIO installed
- USB data cable

For the analysis software:

- Python 3
- Bluetooth Low Energy support
- A Python-compatible BLE adapter
- The dependencies listed below

### 1. Clone the Repository

```bash
git clone https://github.com/BasilAmin/Smart-Sole.git
cd Smart-Sole
```

### 2. Build and Upload the Firmware

Enter the firmware directory:

```bash
cd "SmartSole Firmware"
```

Build the project:

```bash
pio run
```

Upload it to the XIAO ESP32-S3:

```bash
pio run --target upload
```

Open the serial monitor:

```bash
pio device monitor --baud 9600
```

The device should begin advertising over BLE with the name:

```text
Smart Sole
```

### 3. Set Up the Python Environment

Return to the repository root and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the packages used by the Version 1 scripts:

```bash
python -m pip install --upgrade pip
pip install bleak numpy pandas scikit-learn matplotlib tensorflow
```

### 4. Find the Smart Sole BLE Address

With the firmware running, use the BLE scanner:

```bash
python "Smart Sole Software/BLE_ScanForMAC"
```

Locate the device named `Smart Sole` and copy its address.

Replace the hard-coded BLE address in the Python script you intend to run. Depending on your operating system, the device identifier may appear as a MAC address or a platform-specific UUID.

### 5. Receive Raw Sensor Data

From the repository root, run:

```bash
python "Data management/Behaviour_regocnition_BLE.py"
```

This connects to the sole, subscribes to BLE notifications, and parses the three sensor readings.

### 6. Run the Behaviour Model

The main experimental behaviour-detection implementation is located at:

```text
Smart Sole Software/Smart_Sole_Models/Smart_Sole_System_main.py
```

Before running it, replace its machine-specific absolute paths with paths appropriate for your clone. When running from the repository root, the relevant files are:

```text
Smart Sole Software/Behavioural detection model/Behavioural_classification_model_R2.keras
Smart Sole Software/Smart_Sole_Models/Testing_R2_2.csv
```

Then run:

```bash
python "Smart Sole Software/Smart_Sole_Models/Smart_Sole_System_main.py"
```

## Analysis Pipeline

### Behaviour Classification

Sensor samples are collected into a fixed-length buffer, flattened, scaled, and passed to a TensorFlow/Keras classification model.

The Version 1 experiments include behaviour categories such as:

- Standing
- Sitting
- Walking
- Limping
- Stationary heel avoidance
- Dynamic heel avoidance
- Stationary lateral-arch avoidance
- Dynamic lateral-arch avoidance

### Anomaly Detection

The repository also experiments with scikit-learn’s `IsolationForest`.

Reference datasets represent expected readings for different behaviours. A new sample can then be compared with the relevant reference dataset to estimate whether it is anomalous.

This hybrid approach is intended to distinguish between:

1. The activity or behaviour being performed.
2. Whether the pressure pattern appears unusual for that behaviour.

## Version 1 Limitations

Smart Sole V1 is an early research implementation and currently has several limitations:

- Some Python scripts contain hard-coded BLE addresses.
- Some model paths are specific to the original development computer.
- The repository does not yet include a packaged command-line application.
- Python dependencies are not yet pinned in a `requirements.txt`.
- Sensor readings are raw ADC values rather than calibrated pressure measurements.
- The training data is limited and should not be treated as clinically representative.
- Model performance has not been clinically validated.
- Some folders contain earlier or duplicated experiments.
- Automated tests and continuous integration have not yet been added.

## Roadmap

Potential next steps include:

- [ ] Replace hard-coded paths and BLE addresses with a configuration file
- [ ] Add a reproducible `requirements.txt` or `pyproject.toml`
- [ ] Calibrate sensor readings into force or pressure units
- [ ] Add individual user-calibration workflows
- [ ] Expand and document the training dataset
- [ ] Add repeatable model-evaluation metrics
- [ ] Package the Python software into a single application
- [ ] Add automated firmware and software tests
- [ ] Improve the real-time visualisation interface
- [ ] Investigate wearable feedback methods
- [ ] Document hardware assembly and wiring
- [ ] Develop and evaluate a future Version 2 prototype

## Documentation

More information is available in:

- [Smart Sole project page](https://basilamin.com/projects/smart-sole/)
- [Smart Sole Project Book](./Smart%20Sole%20Project%20book.pdf)
- [Smart Sole Presentation Poster](./Smart%20Sole%20Presentation%20Poster.pdf)

## Contributing

Smart Sole is an experimental research project. Issues, technical suggestions, documentation improvements, and pull requests are welcome.

When contributing, please clearly distinguish between:

- Tested functionality
- Experimental functionality
- Proposed medical or clinical applications

Do not present experimental results as medical advice or clinically validated findings.

## License

This project is licensed under the [MIT License](./LICENSE).

## Author

Created by [Basil Amin](https://basilamin.com/).

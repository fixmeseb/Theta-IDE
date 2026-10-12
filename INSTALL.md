# Installation & Setup Guide

This guide walks through setting up the environment to run the training pipeline, testing suites, and backend API service.

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Virtual environment manager (`venv` or `conda`)

---

## 2. Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/CameronEgb/Offline-BlendRL.git
cd Offline-BlendRL
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Install the core framework, development tools and the desktop app into the same venv:

```bash
pip install --upgrade pip
pip install -e ".[dev,gui]"
```

The `gui` extra adds PyQt6, PyQt6-WebEngine, psutil and (on Windows) pywinpty. The desktop app imports the
training pipeline directly, so it must run from this venv; there is no separate frontend venv any more.

*(Optional) Install Atari benchmarks:*
```bash
pip install -e ".[atari]"
```

---

## 3. Verifying Installation

Run the automated test suite to confirm all modules and registries function properly:

```bash
pytest tests/ -v
```

The GUI tests need a display; on a headless machine set `QT_QPA_PLATFORM=offscreen`. Tests marked `slow` train for real and take a few minutes; skip them with `-m "not slow"`.

---

## 4. Launching ThetaIDE

From the repository root, with the venv active:

```bash
python -m frontend
```

That is the only command. There is no server to start first: the app composes and validates configs
in-process with the same code as `run_pipeline.py`, and runs training as a child process (`QProcess`),
streaming its output and metrics into the app through Qt signals.

The experiment builder validates each config as you edit it. The resolved config appears in the
`config.yaml` tab, and the status bar shows `PIPELINE ● local` once the pipeline has loaded. If it shows
`PIPELINE ○ unavailable`, the app is not running from the project venv; reinstall with
`pip install -e ".[gui]"` and start it again with `python -m frontend`.

**Export recipe YAML…** writes a standalone experiment recipe. Save it in `in/config/experiment/thetaide/` and run it with `python run_pipeline.py thetaide/<name>`.

**Launch training** (F5) trains the builder's config on this machine. The Training monitor shows live reward, loss and progress, and the console streams the pipeline's output. **Stop** (Shift+F5) terminates the pipeline and its training processes. Each launch gets a unique experiment ID (`<name>_<YYYYmmdd-HHMMSS>`), so earlier results are never overwritten. Closing the window while a run is training (or the queue is running) hides ThetaIDE to the system tray, and training keeps going; the tray icon shows what is training, notifies you when runs finish, and reopens the window when clicked. **File → Quit** (Ctrl+Q) or **Quit ThetaIDE** in the tray menu ends the app; if a run is still training it asks first, then stops it. The metrics written so far stay in `results/logs/`. Without a system tray, closing the window asks before stopping training instead. **Run → Start simulated demo** works without the training environment.

**Job queue.** **Add to queue** (Ctrl+Shift+Q) adds the builder's config to the queue without starting it. When your list is ready, press **Start queue** in the Queue tab (or Ctrl+Shift+R). Queued experiments then train one at a time, in order, and the Training monitor switches to each one as it starts. **Pause queue** stops the next job from starting; one already training keeps running. The queue pauses itself when it runs out of jobs. The **Queue** tab lists the running, queued and recent jobs, with buttons to move a queued job up or down, remove it, or open it in the monitor. Queued jobs are saved with the workspace and are still waiting the next time you open ThetaIDE.

**TensorBoard tab.** Runs launched with **Log to TensorBoard**, which is on by default, also write TensorBoard logs to `results/tensorboard/`. **Start server** in the TensorBoard tab runs a local TensorBoard server (bound to 127.0.0.1) over that folder; **Open in browser** opens it once it is serving, and **Stop server** ends it. ThetaIDE stops it when it closes.

Run records are stored in `.thetaide/runs/` by default; override this with `--data-dir <path>`.

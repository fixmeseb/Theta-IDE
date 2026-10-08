"""Rich documentation data and articles covering Theta-IDE, a general-purpose ML, DL, and RL experimentation platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DocArticle:
    id: str
    title: str
    category: str
    summary: str
    html_content: str
    keywords: List[str]


ARTICLES: List[DocArticle] = [
    DocArticle(
        id="overview",
        title="Welcome & Architecture Overview",
        category="Getting Started",
        summary="High-level architecture of Theta-IDE, the execution pipeline, and general-purpose ML/DL/RL experimentation.",
        keywords=["overview", "architecture", "intro", "welcome", "reinforcement learning", "deep learning", "machine learning"],
        html_content="""
<h2>Welcome to Theta-IDE</h2>
<p><b>Theta-IDE</b> is a general-purpose development environment and experimentation platform for <b>Reinforcement Learning (RL)</b>, <b>Deep Learning (DL)</b>, <b>Machine Learning (ML)</b>, and <b>modular algorithm benchmarking</b>.</p>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Core Philosophy:</b> Unify the entire research experimentation lifecycle into a single, cohesive workflow &mdash; from visual Hydra configuration composition and real-time training telemetry, to Slurm cluster job orchestration, interactive visualization analysis, and an extensible Component Hub.
</div>

<h3>Framework Architecture</h3>
<ul>
  <li><b>Hydra Configuration Layer (<code>in/config/</code>):</b> Declarative, three-tier hierarchical configuration defining environments, agents, models, cluster resources, and experiment recipes.</li>
  <li><b>FastAPI Backend Daemon (<code>src/app/api/</code>):</b> Decoupled API service managing local training subprocesses, Slurm cluster submissions, and job queue states.</li>
  <li><b>PyTorch Lightning Runtime (<code>src/app/train.py</code>):</b> Standardized training driver providing automated checkpointing, device acceleration (CUDA/MPS/CPU), and structured logging.</li>
  <li><b>Modular Model &amp; Method Architecture:</b> Pluggable registries for RL algorithms and model architectures. Theta-IDE ships with the <b>PPO</b>, <b>CQL</b> and <b>IQL</b> algorithms and with MLP, CNN, Dueling ResNet, Transformer, Cross-Attention and CEW models, plus the composite BlendRL model. Other algorithms and architectures can be added as plugins (see <a href="authoring_plugins">Authoring Guide</a>).</li>
  <li><b>Extensible Component Hub &amp; Plugin System:</b> Core plugins shipping natively with zero overhead when toggled off, plus community plugins (UI tabs, RL methods, models, environments, and experiment recipes) installable directly from the Community Hub.</li>
</ul>

<h3>Key Workflows</h3>
<ol>
  <li><b>Configure:</b> Select or compose an experiment in the <i>Experiment Config</i> pane.</li>
  <li><b>Execute:</b> Train directly on your local machine or submit Slurm batch jobs to an HPC cluster.</li>
  <li><b>Monitor:</b> Observe real-time reward curves, minibatch losses, and metric tables with moving-average curve smoothing.</li>
  <li><b>Analyze:</b> Generate publication-ready convergence plots, loss breakdowns, and markdown comparative reports.</li>
</ol>
""",
    ),
    DocArticle(
        id="launching",
        title="Installing & Launching",
        category="Getting Started",
        summary="Set up the backend and frontend environments, then start Theta-IDE on macOS, Linux, or Windows.",
        keywords=["install", "setup", "launch", "start", "run", "venv", "uvicorn", "backend", "windows", "powershell", "api url"],
        html_content=r"""
<h2>Installing & Launching</h2>
<p>Theta-IDE has two parts, each with its own virtual environment. Run every command from the <b>repository root</b>.</p>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Part</td>
    <td>What it does</td>
    <td>Environment</td>
  </tr>
  <tr><td><b>Backend API</b></td><td>Validates configs, runs training, serves TensorBoard</td><td><code>venv/</code></td></tr>
  <tr><td><b>Frontend</b></td><td>This desktop app</td><td><code>frontend/.venv/</code> (PyQt6 only)</td></tr>
</table>

<h3>1. One-Time Setup</h3>
<p><b>macOS / Linux:</b></p>
<pre><code>python3 -m venv venv
source venv/bin/activate
pip install -e ".[api]"
deactivate

python3 -m venv frontend/.venv
source frontend/.venv/bin/activate
pip install -r frontend/requirements.txt</code></pre>

<p><b>Windows (PowerShell):</b></p>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, so venvs can activate

py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[api]"
deactivate

py -m venv frontend\.venv
.\frontend\.venv\Scripts\Activate.ps1
pip install -r frontend\requirements.txt</code></pre>

<h3>2. Start the Backend (Terminal 1)</h3>
<pre><code>source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000</code></pre>
<p>Leave it running. Its interactive API docs are at <code>http://127.0.0.1:8000/docs</code>.</p>

<h3>3. Start the Frontend (Terminal 2)</h3>
<pre><code>source frontend/.venv/bin/activate   # Windows: .\frontend\.venv\Scripts\Activate.ps1
python -m frontend</code></pre>

<h3>Launch Options</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Option</td>
    <td>Default</td>
    <td>Purpose</td>
  </tr>
  <tr><td><code>--api-url &lt;url&gt;</code></td><td><code>http://127.0.0.1:8000</code></td><td>Backend address. The <code>THETAIDE_API_URL</code> environment variable sets it too.</td></tr>
  <tr><td><code>--data-dir &lt;path&gt;</code></td><td><code>.thetaide/runs/</code></td><td>Where run records are stored.</td></tr>
  <tr><td><code>--install-desktop-entry</code></td><td></td><td><b>Linux only:</b> add Theta-IDE to the application menu, then exit.</td></tr>
</table>

<h3>Reloading While Developing</h3>
<p>Press <code>Ctrl+R</code> in the terminal that launched the app to restart it with your latest code.</p>
<ul>
  <li><b>macOS / Linux:</b> works repeatedly.</li>
  <li><b>Windows:</b> works once per launch. Windows can't replace a running program in place, so the reload starts a fresh copy and the terminal prompt returns; the new window no longer listens to the terminal. It also needs a real console (PowerShell, Windows Terminal, or Command Prompt), not Git Bash.</li>
</ul>

<h3>Without the Backend</h3>
<p>The app still opens. You can browse and edit configs (the preview shows an unvalidated local draft) and run <i>Run &rarr; Start simulated demo</i>. Launching training, the job queue, and the TensorBoard tab need the backend.</p>

<h3>Stopping</h3>
<ul>
  <li><b>Frontend:</b> close the window. Training that is already running keeps going, and reopening the app reconnects to it.</li>
  <li><b>Backend:</b> press <code>Ctrl+C</code> in its terminal. This also stops TensorBoard.</li>
</ul>
""",
    ),
    DocArticle(
        id="managing_experiments",
        title="Creating, Renaming & Deleting Experiments",
        category="Getting Started",
        summary="Create new experiments and groups, rename or delete configs, and save or export your changes.",
        keywords=["new", "create", "rename", "delete", "duplicate", "experiment", "group", "save", "export", "config tree"],
        html_content=r"""
<h2>Creating, Renaming & Deleting Experiments</h2>
<p>Experiments live in <code>in/config/experiment/&lt;group&gt;/&lt;name&gt;.yaml</code>. Each group has a <code>_base.yaml</code> that sets the environment and paradigm its experiments inherit. Manage them from the config tree in the <a href="ide://config">Experiment Config</a> pane.</p>

<h3>Creating an Experiment</h3>
<ol>
  <li>Choose <i>File &rarr; New experiment</i> (<code>Ctrl+N</code>), or right-click a group and choose <b>New Experiment in this group&hellip;</b>.</li>
  <li>In the <b>New Experiment</b> dialog, pick a group and enter a name.</li>
  <li>To start a <b>new group</b>, type a group name that doesn't exist yet. The dialog then asks for the <b>paradigm</b> and <b>environment</b>, and creates the group's <code>_base.yaml</code> for you. Only environments compatible with the chosen paradigm are offered.</li>
</ol>
<p>Right-click a component category and choose <b>New Component in this category&hellip;</b> to add an agent, model, or environment config the same way.</p>

<h3>Right-Click Menu</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Action</td>
    <td>What it does</td>
  </tr>
  <tr><td><b>Load in Config Viewer</b></td><td>Open the file in the form editor</td></tr>
  <tr><td><b>Rename&hellip;</b></td><td>Rename the file. The <code>experiment_id</code> inside it is updated to match.</td></tr>
  <tr><td><b>Delete&hellip;</b></td><td>Delete the file after confirmation</td></tr>
  <tr><td><b>New Experiment in this group&hellip;</b></td><td>Create an experiment in the selected group</td></tr>
  <tr><td><b>New Component in this category&hellip;</b></td><td>Create an agent, model, or environment config</td></tr>
  <tr><td><b>Collapse All / Expand All</b></td><td>Fold or unfold the tree</td></tr>
</table>

<div style="background-color: rgba(251, 73, 52, 0.12); border-left: 4px solid #fb4934; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Protected base files:</b> A group's <code>_base.yaml</code> can't be deleted while experiments in that group still inherit from it, because Hydra could no longer load them. Delete or move those experiments first.
</div>
<p>Deleted files are removed from your working tree. If the file was committed to git, you can restore it with <code>git checkout -- &lt;path&gt;</code>.</p>

<h3>Saving & Exporting</h3>
<ul>
  <li><b>Save configuration</b> (<code>Ctrl+S</code>): write your edits back to the experiment's YAML file.</li>
  <li><b>Export draft YAML</b> (<code>Ctrl+Shift+S</code>): write a standalone recipe. Save it in <code>in/config/experiment/thetaide/</code> and run it with <code>python run_pipeline.py thetaide/&lt;name&gt;</code>.</li>
</ul>
""",
    ),
    DocArticle(
        id="ui_panels",
        title="IDE Panels & Navigation Guide",
        category="Interface",
        summary="Tour of all ten sidebar panels, from Components and Config to the Terminal and Console.",
        keywords=["panels", "ui", "navigation", "sidebar", "tabs", "interface", "layout"],
        html_content="""
<h2>IDE Panels & Navigation</h2>
<p>Theta-IDE organizes its interface into vertical sidebar tabs. Each tab represents a specialized workstation for your experiments.</p>
<p>Results, Plots, TensorBoard and Queue are hidden by default. Turn them on in <i>Settings &rarr; Appearance</i> (see <a href="sidebar_layout">Sidebar &amp; Layout</a>).</p>

<h3>1. Experiment Config (<code>config</code>)</h3>
<p>The primary workspace for configuring and launching experiments:</p>
<ul>
  <li><b>Config Tree:</b> Hierarchical browser mirroring <code>in/config/experiment/</code>. Browse by group (e.g. <i>cartpole</i>, <i>mimic</i>, <i>quick_tests</i>).</li>
  <li><b>Boxed Config Viewer:</b> Visual, form-driven editor for hyperparameters, paradigms, learning rates, epochs, and Slurm resource allocations.</li>
  <li><b>Hydra YAML Preview:</b> Real-time preview of the fully resolved Hydra configuration with syntax highlighting and validation errors.</li>
  <li><b>Action Bar:</b> Launch Training (<code>F5</code>), Add to Batch Queue (<code>Ctrl+Shift+Q</code>), Save YAML (<code>Ctrl+S</code>), Duplicate Config, and Export Recipe.</li>
</ul>

<h3>2. Training Monitor (<code>monitor</code>)</h3>
<p>Real-time telemetry and telemetry replay for live and finished training runs:</p>
<ul>
  <li><b>Metric Cards:</b> Key scalars including Episode Reward, Minibatch Loss, and Timestep Budget.</li>
  <li><b>Dynamic Curves:</b> Live matplotlib charts tracking reward and loss trajectories.</li>
  <li><b>Smoothing Slider:</b> Exponential Moving Average (EMA) smoothing from 0% (raw points) up to 95% (smoothed trends).</li>
  <li><b>Multi-Run Navigation:</b> Previous/Next buttons and dropdown selector to compare against previous runs in history.</li>
  <li><b>Pin as Baseline:</b> Pin any historic run as a dashed reference curve overlay on top of active runs.</li>
  <li><b>Jump to Live:</b> Instant one-click camera focus back to the actively training experiment.</li>
</ul>

<h3>3. Job Queue (<code>queue</code>)</h3>
<p>Line up several experiments and train them one after another:</p>
<ul>
  <li>Add the loaded experiment with <b>Run &rarr; Add to queue</b> (<code>Ctrl+Shift+Q</code>). Adding does not start it.</li>
  <li><b>Start queue</b> (<code>Ctrl+Shift+R</code>) trains queued experiments one at a time, in order. The Training Monitor switches to each one as it starts.</li>
  <li><b>Pause queue</b> stops the next job from starting; a job already training keeps running. The queue also pauses itself when it runs out of jobs.</li>
  <li>The list shows running, queued and recent jobs. Select a job to <b>Move up</b>, <b>Move down</b>, <b>Remove from queue</b>, or <b>Open in monitor</b>.</li>
  <li>The queue is kept by the backend, so it continues while Theta-IDE is closed, but it is lost if the backend restarts.</li>
</ul>

<h3>4. Plots & Visualizations (<code>plots</code>)</h3>
<p>Inspect output plots auto-generated at the end of training pipelines:</p>
<ul>
  <li><b>Convergence Curves:</b> Visualizes mean ± SEM evaluation rewards across seeds and algorithms.</li>
  <li><b>Loss Decomposition:</b> Explores actor loss, critic/Q-loss, policy entropy, and Bellman error over transitions.</li>
  <li><b>Report Generator:</b> Markdown comparison tables showing best metrics and hyperparameter configurations.</li>
</ul>

<h3>5. Embedded Terminal (<code>terminal</code>)</h3>
<p>Built-in interactive terminal powered by xterm.js:</p>
<ul>
  <li>Full shell access with support for zsh, bash, and tmux.</li>
  <li>Terminal precedence mode: allows tmux prefix keys and terminal hotkeys to pass through uninterrupted.</li>
  <li><b>macOS and Linux only.</b> The terminal needs a POSIX pseudo-terminal, so it is unavailable on Windows. Use a separate PowerShell window there.</li>
</ul>

<h3>6. Modular Components &amp; Community Hub (<code>components</code>)</h3>
<p>Manage modular algorithm building blocks and browse community extensions:</p>
<ul>
  <li><b>Components Tree &amp; Viewer:</b> Browse and edit modular configurations outside experiments (<code>agent</code> profiles, <code>env</code> definitions, <code>model</code> architectures, <code>paradigms</code>, and <code>site</code> profiles).</li>
  <li><b>Community Hub Dialog:</b> Marketplace modal to discover, install, update, and uninstall community plugins, RL methods, models, and environments directly into your workspace.</li>
</ul>

<h3>7. Workflows (<code>workflows</code>)</h3>
<p>Compose multi-stage experiments as string diagrams, wiring the outputs of one stage into the inputs of the next. See <a href="workflows_pane">Workflows Pane</a>.</p>

<h3>8. Results Browser (<code>results</code>)</h3>
<p>Every run you have launched, with its seed, status and reward. Reload a run's config, compare two runs, or open a run's plots. See <a href="results_browser">Results Browser</a>.</p>

<h3>9. TensorBoard (<code>tensorboard</code>)</h3>
<p>An embedded TensorBoard dashboard for runs logged with <i>Log to TensorBoard</i>. See <a href="tensorboard_tab">TensorBoard Tab</a>.</p>

<h3>10. Console (<code>console</code>)</h3>
<p>Streams the pipeline's output and accepts a few quick commands. See <a href="console_pane">Console</a>.</p>
""",
    ),
    DocArticle(
        id="learning_paradigms",
        title="Learning Paradigms & Constraints",
        category="Machine Learning",
        summary="Explanation of online RL, offline RL, and supervised learning paradigms supported in Theta-IDE.",
        keywords=["paradigms", "online", "offline", "supervised", "ppo", "cql", "iql", "sac", "dataset"],
        html_content="""
<h2>Learning Paradigms</h2>
<p>Theta-IDE strictly validates experiment compatibility using declared <b>paradigms</b>. Every experiment declares <code>paradigm: &lt;name&gt;</code>, which configures the training driver, callbacks, and validation rules.</p>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Paradigm</td>
    <td>Allowed Agents / Models</td>
    <td>Evaluation Mechanism</td>
    <td>Key Constraints</td>
  </tr>
  <tr>
    <td><b><code>online_rl</code></b></td>
    <td>Online RL algorithms (PPO; others via plugins)</td>
    <td>Fixed-episode live simulator rollouts via <code>EnvironmentEvaluatorCallback</code></td>
    <td><code>offline_only: false</code>; generates transition datasets</td>
  </tr>
  <tr>
    <td><b><code>offline_rl</code></b></td>
    <td>Offline RL algorithms (CQL, IQL; others via plugins)</td>
    <td>Replay buffer data module; validation loss and Bellman error via Lightning</td>
    <td><code>intervals_count: 1</code>, <code>eval_episodes: 0</code>; requires static transition dataset</td>
  </tr>
  <tr>
    <td><b><code>supervised</code></b></td>
    <td>Predictive architectures (DNN, CNN, Dueling ResNet, Transformer, CEW)</td>
    <td>Validation cross-entropy / MSE / AUROC on held-out splits</td>
    <td>Standalone RL agents forbidden</td>
  </tr>
</table>

<h3>Transition Dataset Schema</h3>
<p>When online agents collect experience or offline datasets are loaded into replay buffers, transitions adhere to standard Gym / Gymnasium conventions:</p>

<h4>1. Core Standard RL Fields (Required &amp; Native)</h4>
<p>All standard Gym / Gymnasium environments and baseline offline datasets produce and consume the clean standard RL 5-tuple:</p>
<pre><code>{
  "obs": np.ndarray,            # Primary observation vector or image tensor
  "action": int | np.ndarray,   # Selected discrete action index or continuous action vector
  "reward": float,              # Scalar transition reward
  "next_obs": np.ndarray,       # Subsequent observation state
  "done": bool                  # Episode termination flag
}</code></pre>

<h4>2. Optional Domain-Specific Extensions</h4>
<p>Theta-IDE's open architecture allows community plugins and custom models (such as neurosymbolic hybrids, goal-conditioned agents, or multi-modal policies) to store or consume optional auxiliary fields via dataset adapters:</p>
<pre><code>{
  "logic_obs": np.ndarray,      # Optional: symbolic facts for logic reasoner plugins (e.g. BlendRL)
  "next_logic_obs": np.ndarray, # Optional: subsequent symbolic facts
  "info": dict                  # Optional: environment metadata or diagnostic flags
}</code></pre>

<div style="background-color: rgba(69, 133, 136, 0.12); border-left: 4px solid #458588; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Standard Environments vs. Plugin Extensions:</b>
  <ul style="margin: 6px 0 0 0; padding-left: 18px;">
    <li><b>Standard environments do NOT emit domain-specific logic natively:</b> Gym, Gymnasium, and standard benchmark environments produce clean observation vectors or images.</li>
    <li><b>Pure neural algorithms:</b> Baseline algorithms like PPO, IQL, and CQL train strictly on the core 5 fields and completely ignore auxiliary fields.</li>
    <li><b>Extensible adapters:</b> Specialized plugins (such as hybrid reasoners) compute relational groundings or custom representations on the fly via their own internal wrappers, leaving the core dataset schema clean and universal.</li>
  </ul>
</div>

<p>Online dataset generation via <code>DatasetWriter</code> automatically writes chunked <code>.pkl</code> archives (typically 100,000 transitions per chunk) accompanied by a <code>dataset_manifest.json</code> capturing git provenance, random seed, transition count, and environment metadata.</p>
""",
    ),
    DocArticle(
        id="blendrl_hybrid",
        title="Plugin Case Study: BlendRL Neurosymbolic Policy",
        category="Machine Learning",
        summary="A case study demonstrating how composite models and first-order logic reasoners integrate into Theta-IDE via the Component Plugin system.",
        keywords=["blendrl", "symbolic", "nsfr", "neumann", "neural", "prolog", "logic", "hybrid", "plugin", "component"],
        html_content="""
<h2>Plugin Case Study: BlendRL Neurosymbolic Architecture</h2>
<p>Theta-IDE is designed to support any machine learning, deep learning, or reinforcement learning architecture. To illustrate how specialized research algorithms integrate into the IDE without core modification, this article examines <b>BlendRL</b> &mdash; a composite model plugin available on the <b>Community Hub</b>.</p>

<div style="background-color: rgba(184, 187, 38, 0.12); border-left: 4px solid #b8bb26; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Modular Extension in Action:</b> Rather than hardcoding specialized reasoning logic into Theta-IDE, BlendRL is packaged as an installable model and method component. It bridges standard deep neural policies with first-order symbolic logic reasoners (such as NSFR and Neumann) through an adaptive blending module.
</div>

<h3>Composite Architecture Overview</h3>
<p>As a composite plugin, BlendRL assembles three constituent modules defined in Hydra:</p>
<ol>
  <li><b>Neural Encoder (<code>neural</code>):</b> Multi-Layer Perceptrons (MLP), Dueling ResNets, or Transformers that process raw continuous or pixel observations.</li>
  <li><b>Symbolic Reasoner (<code>symbolic</code>):</b>
    <ul>
      <li><b>NSFR (Neural Symbolic Forward Reasoner):</b> Differentiable forward-chaining deduction engine operating on grounded facts and rules.</li>
      <li><b>Neumann Reasoner:</b> Matrix-based forward reasoner designed for accelerated rule valuation.</li>
    </ul>
  </li>
  <li><b>The Blender (<code>blender</code>):</b> Combines neural logits \\(\\pi_{neural}(a|s)\\) and symbolic valuation scores \\(\\pi_{logic}(a|s)\\):
    <pre><code>\\pi_{blended}(a|s) = (1 - \\alpha) \\cdot \\pi_{neural}(a|s) + \\alpha \\cdot \\pi_{logic}(a|s)</code></pre>
    where \\(\\alpha\\) can be fixed, learned, or dynamically gated by symbolic confidence.
  </li>
</ol>

<h3>Decoupled State Valuation</h3>
<p>Because native Gym environments do not output logic representations, the BlendRL plugin handles grounding internally:</p>
<ul>
  <li>If optional precomputed <code>logic_obs</code> exist in a custom transition dataset, they are utilized directly.</li>
  <li>Otherwise, the plugin agent's <code>_prepare_logic_obs()</code> automatically grounds raw continuous/discrete state vectors into relational predicates on the fly, evaluating them against domain rules without requiring special environment wrappers.</li>
</ul>

<h3>Key Takeaway for Component Authors</h3>
<p>BlendRL serves as a blueprint for researchers building custom models in Theta-IDE: whether developing diffusion-based policies, hierarchical RL agents, or symbolic reasoners, authors can encapsulate their architectures into self-contained plugins that install cleanly through the <b>Community Hub</b>.</p>
""",
    ),
    DocArticle(
        id="hotkeys",
        title="Keyboard Shortcuts & Leader Chords",
        category="Workflow & Tools",
        summary="Complete reference for tmux-style leader key navigation and quick action hotkeys.",
        keywords=["hotkeys", "shortcuts", "leader", "tmux", "keyboard", "navigation"],
        html_content="""
<h2>Keyboard Shortcuts & Leader Navigation</h2>
<p>Theta-IDE features a high-efficiency <b>Leader key system</b> inspired by tmux and Vim. You can navigate between any panel instantly without taking your hands off the keyboard.</p>

<h3>The Action Key (Leader)</h3>
<p>Default: <code>Ctrl+B</code> (Customizable in <i>Settings &rarr; Hotkeys</i> to <code>Caps Lock</code>, <code>Alt</code>, <code>Ctrl</code>, or <code>Meta</code>).</p>
<p>Supports two operational modes:</p>
<ul>
  <li><b>Leader (Modal):</b> Tap the Action key, release it, and press a digit within 1.5 seconds.</li>
  <li><b>Chorded:</b> Hold the Action key and tap a digit simultaneously.</li>
</ul>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Target Pane / Action</td>
  </tr>
  <tr>
    <td><code>Action + 0</code></td>
    <td>Settings & Preferences</td>
  </tr>
  <tr>
    <td><code>Action + 1</code></td>
    <td>Components (configs and Community Hub)</td>
  </tr>
  <tr>
    <td><code>Action + 2</code></td>
    <td>Experiment Configuration (Editor)</td>
  </tr>
  <tr>
    <td><code>Action + 3</code></td>
    <td>Training Monitor & Curves</td>
  </tr>
  <tr>
    <td><code>Action + 4</code></td>
    <td>Results Browser</td>
  </tr>
  <tr>
    <td><code>Action + 5</code></td>
    <td>Plot & Visualization Viewer</td>
  </tr>
  <tr>
    <td><code>Action + 6</code></td>
    <td>TensorBoard Dashboard</td>
  </tr>
  <tr>
    <td><code>Action + 7</code></td>
    <td>Job Queue Manager</td>
  </tr>
  <tr>
    <td><code>Action + 8</code></td>
    <td>Interactive Terminal</td>
  </tr>
  <tr>
    <td><code>Action + 9</code></td>
    <td>System Console Log</td>
  </tr>
</table>

<p>The Workflows pane has no digit by default. Reach it from <i>View &rarr; Workflows</i>, or remap a digit in <i>Settings &rarr; Hotkeys &rarr; Pane Navigation Hotkey Menu</i>.</p>

<h3>Menu Shortcuts</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Shortcut</td>
    <td>Action</td>
  </tr>
  <tr><td><code>F5</code></td><td>Run &rarr; Launch training with the loaded experiment</td></tr>
  <tr><td><code>Shift+F5</code></td><td>Run &rarr; Stop the active run</td></tr>
  <tr><td><code>Ctrl+F5</code></td><td>Run &rarr; Start simulated demo (no backend needed)</td></tr>
  <tr><td><code>Ctrl+Shift+Q</code></td><td>Run &rarr; Add to queue</td></tr>
  <tr><td><code>Ctrl+Shift+R</code></td><td>Run &rarr; Start or pause queue</td></tr>
  <tr><td><code>Ctrl+N</code></td><td>File &rarr; New experiment</td></tr>
  <tr><td><code>Ctrl+S</code></td><td>File &rarr; Save configuration</td></tr>
  <tr><td><code>Ctrl+Shift+S</code></td><td>File &rarr; Export draft YAML</td></tr>
  <tr><td><code>Ctrl+Q</code></td><td>File &rarr; Quit</td></tr>
</table>

<h3>Changing Shortcuts</h3>
<ul>
  <li>Rebind menu shortcuts in <i>Settings &rarr; Hotkeys &rarr; Application Menu Shortcuts</i>, or under <code>[shortcuts]</code> in <code>settings.toml</code> (for example <code>launch_training = "F6"</code>).</li>
  <li>Change the Action key, leader timeout and digit-to-pane mapping under <code>[hotkeys]</code>.</li>
</ul>

<h3>Reloading the App (Developers)</h3>
<p><code>Ctrl+R</code> is not an in-app shortcut. Pressed in the <b>terminal that launched Theta-IDE</b>, it restarts the app with your latest code changes. See <a href="launching">Installing &amp; Launching</a>.</p>
""",
    ),
    DocArticle(
        id="config_system",
        title="3-Tier Hierarchical Configuration",
        category="Getting Started",
        summary="How Hydra configuration files are structured across Tier 1 (Defaults), Tier 2 (Universal), and Tier 3 (Methods).",
        keywords=["config", "hydra", "tier", "yaml", "methods", "params", "hyperparameters"],
        html_content=r"""
<h2>3-Tier Hierarchical Configuration</h2>
<p>Configurations in Theta-IDE use a strict 3-tier hierarchy that eliminates parameter duplication while allowing granular per-method overrides.</p>

<h3>Tier 1: Defaults & Base Profiles</h3>
<p>Stored under <code>in/config/agent/&lt;algo&gt;.yaml</code> and <code>in/config/model/&lt;arch&gt;.yaml</code>. These define the baseline algorithm and model parameters (e.g. default batch size, discount factor \(\gamma\), network layer dimensions).</p>

<h3>Tier 2: Universal Experiment Parameters (<code>methods.params</code>)</h3>
<p>Universal scalars and hyperparameters applied across all methods in a single experiment:</p>
<pre><code>methods:
  params:
    epochs_per_interval: 25
    gamma: 0.99
    lr: 3e-4
    agent:
      cql:
        cql_alpha: 5.0
    model:
      dueling_resnet:
        hidden_dim: 128</code></pre>

<h3>Tier 3: Method-Level Declarations</h3>
<p>Specific methods declared for execution inherit from Tier 1 and Tier 2, specifying only their differences:</p>
<pre><code>methods:
  cql_baseline:
    agent: cql
    model: dnn
  ppo_transformer:
    agent: ppo
    model: transformer
  cql_blendrl_hybrid: # Optional composite model plugin
    agent: cql
    model:
      blendrl:
        neural: dueling_resnet
        symbolic:
          nsfr:
            ruleset: cartpole_rules</code></pre>

<h3>Environment Keys</h3>
<p>Environment YAMLs (<code>in/config/env/*.yaml</code>) define operational metadata declaratively:</p>
<ul>
  <li><code>offline_only: true | false</code> &mdash; drives paradigm verification</li>
  <li><code>monitor_metric: "eval/reward" | "val/loss"</code> &mdash; target metric for checkpointing and tuning</li>
  <li><code>preprocess_on_load: true | false</code> &mdash; whether dataset requires offline conversion</li>
  <li><code>default_plots: [...]</code> &mdash; default visualizers auto-run after training</li>
</ul>
""",
    ),
    DocArticle(
        id="cluster_slurm",
        title="Cluster Execution & Slurm Runner",
        category="Workflow & Tools",
        summary="How to submit jobs to HPC clusters using Slurm site profiles, resource limits, and email notifications.",
        keywords=["slurm", "cluster", "hpc", "ncshare", "arc", "sbatch", "gpu"],
        html_content="""
<h2>Cluster & Slurm Execution</h2>
<p>Theta-IDE supports seamless transitions between local testing and cluster-scale execution via Slurm.</p>

<h3>Site Profiles (<code>site</code>)</h3>
<ul>
  <li><code>site=local</code> (default): Interactive local execution inside subprocesses.</li>
  <li><code>site=ncshare</code> / <code>site=arc</code>: Automatically generates and dispatches Slurm batch scripts (<code>sbatch</code>).</li>
</ul>

<div style="background-color: rgba(251, 73, 52, 0.12); border-left: 4px solid #fb4934; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Cluster Push Mandate:</b> Always commit and push all code changes to GitHub before submitting cluster jobs, as remote compute nodes sync directly from the repository.
</div>

<h3>Configuring Cluster Resources</h3>
<p>Specify compute requirements directly in your experiment YAML:</p>
<pre><code>resources:
  time: "04:00:00"   # Wall clock limit (hh:mm:ss)
  gpus: 1            # Number of GPU accelerators
  cores: 16          # CPU cores allocated
  memory: "32G"      # RAM allocation</code></pre>
<p>Priority order: CLI flags &gt; experiment YAML &gt; site default config &gt; fallback defaults.</p>

<h3>Automated Slurm Pipeline Features</h3>
<ul>
  <li><b>Dependency Chaining:</b> Online data generation jobs automatically establish Slurm dependency holds (<code>--dependency=afterok:&lt;job_id&gt;</code>) on downstream offline comparison jobs.</li>
  <li><b>Status Notifications:</b> Configured with <code>--mail-type=END,FAIL</code> for immediate progress updates.</li>
</ul>
""",
    ),
    DocArticle(
        id="plugins_guide",
        title="Core Plugins & Community Extensions",
        category="Extensibility",
        summary="Guide to creating and managing plugins: Core vs Community plugins, lifecycle hooks, and the Plugin Context API.",
        keywords=["plugins", "extensions", "core", "community", "lifecycle", "api", "hub"],
        html_content="""
<h2>Plugin Architecture & Extensibility</h2>
<p>Theta-IDE features a modular plugin architecture modeled on Obsidian's core/community plugin design.</p>

<h3>Core Plugins vs. Community Plugins</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Aspect</td>
    <td>Core Plugins (e.g. Documentation)</td>
    <td>Community Plugins</td>
  </tr>
  <tr>
    <td><b>Origin</b></td>
    <td>Shipped natively with Theta-IDE source repository (<code>frontend/plugins/core/</code>)</td>
    <td>Installed from Community Hub into user storage (<code>.thetaide/plugins/</code>)</td>
  </tr>
  <tr>
    <td><b>Uninstallable</b></td>
    <td><b>No</b> &mdash; core features cannot be deleted from the filesystem</td>
    <td><b>Yes</b> &mdash; can be uninstalled and removed via trash button</td>
  </tr>
  <tr>
    <td><b>System Impact</b></td>
    <td><b>Zero impact when toggled off</b> &mdash; all panes, listeners, and widgets are cleanly unmounted</td>
    <td>Zero impact when toggled off</td>
  </tr>
  <tr>
    <td><b>Settings Location</b></td>
    <td><i>Settings &rarr; Core Plugins</i></td>
    <td><i>Settings &rarr; Community Plugins</i></td>
  </tr>
</table>

<h3>Plugin Structure</h3>
<p>Every plugin requires a directory containing:</p>
<ol>
  <li><code>plugin.json</code>: Metadata manifest (ID, name, description, version, entry point, <code>core: true/false</code>).</li>
  <li><code>__init__.py</code>: Exports a class subclassing <code>Plugin</code>.</li>
</ol>

<h3>Plugin Lifecycle Hooks</h3>
<pre><code>from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

class MyPlugin(Plugin):
    def activate(self, context: PluginContext) -> None:
        # Called when the plugin is enabled
        self.context = context
        self.widget = MyCustomWidget()
        context.add_sidebar_tab(
            tab_id="my_plugin",
            widget=self.widget,
            title="My Extension",
            icon_name="my_icon",
            short_label="Extension"
        )

    def deactivate(self) -> None:
        # Called when toggled off or during IDE shutdown
        # Must clean up all widgets and event listeners
        if self.context:
            self.context.remove_sidebar_tab("my_plugin")
            self.widget.deleteLater()
            self.widget = None
            self.context = None</code></pre>
""",
    ),
    DocArticle(
        id="authoring_plugins",
        title="Authoring Guide: All Component & Plugin Types",
        category="Extensibility",
        summary="Comprehensive developer guide for authoring, registering, and packaging UI Plugins, RL Methods, Models, Environments, and Experiment Recipes.",
        keywords=[
            "authoring", "developer", "plugin", "method", "model", "env", "experiment",
            "component", "hub", "packaging", "protocols", "register_agent", "register_model"
        ],
        html_content="""
<h2>Authoring Guide: All Component & Plugin Types</h2>
<p>Theta-IDE features a modular architecture where nearly every capability &mdash; from UI tabs to RL training algorithms, neural-symbolic models, and benchmark environments &mdash; can be developed, tested, and distributed as an installable component plugin via the <b>Community Hub</b>.</p>

<div style="background-color: rgba(69, 133, 136, 0.12); border-left: 4px solid #83a598; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Five Modular Component Types:</b> Theta-IDE distinguishes between 5 distinct component kinds. Each kind has its own standard destination path, configuration target, and registration mechanism.
</div>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Kind</td>
    <td>Target Directory</td>
    <td>Config Location</td>
    <td>Registration Mechanism</td>
  </tr>
  <tr>
    <td><b><code>plugin</code></b></td>
    <td><code>.thetaide/plugins/&lt;id&gt;/</code></td>
    <td>N/A (Managed by settings)</td>
    <td><code>PluginManager</code> scans <code>plugin.json</code></td>
  </tr>
  <tr>
    <td><b><code>method</code></b></td>
    <td><code>src/usr/methods/&lt;id&gt;/</code></td>
    <td><code>in/config/agent/&lt;id&gt;.yaml</code></td>
    <td><code>@register_agent("&lt;prefix&gt;")</code> from <code>src/usr/methods/agent_registry.py</code></td>
  </tr>
  <tr>
    <td><b><code>model</code></b></td>
    <td><code>src/usr/models/&lt;id&gt;/</code></td>
    <td><code>in/config/model/&lt;id&gt;.yaml</code></td>
    <td><code>@register_model("&lt;name&gt;")</code> in <code>model_registry.py</code></td>
  </tr>
  <tr>
    <td><b><code>env</code></b></td>
    <td><code>in/envs/&lt;id&gt;/</code></td>
    <td><code>in/config/env/&lt;id&gt;.yaml</code></td>
    <td><code>VectorizedBaseEnv.from_name()</code> factory</td>
  </tr>
  <tr>
    <td><b><code>experiment</code></b></td>
    <td><code>in/config/experiment/&lt;id&gt;/</code></td>
    <td><code>in/config/experiment/&lt;id&gt;.yaml</code></td>
    <td>Dispatched via <code>run_pipeline.py</code></td>
  </tr>
</table>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>1. Authoring UI / Frontend Plugins (<code>kind: "plugin"</code>)</h3>
<p>UI plugins extend the Theta-IDE graphical interface by adding custom sidebar workstation tabs, status bar telemetry, or background tools.</p>

<h4>Directory Structure</h4>
<pre><code>my_tool/
├── plugin.json       # Manifest metadata
└── __init__.py       # Plugin class entry point</code></pre>

<h4>Manifest (<code>plugin.json</code>)</h4>
<pre><code>{
  "id": "my_tool",
  "name": "My Custom Tool",
  "version": "1.0.0",
  "description": "Interactive analysis tool for reinforcement learning checkpoints.",
  "author": "Your Name",
  "default_enabled": false,
  "icon": "terminal",
  "kind": "plugin",
  "entry_point": "MyToolPlugin"
}</code></pre>

<h4>Python Implementation (<code>__init__.py</code>)</h4>
<pre><code>from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

class MyToolPlugin(Plugin):
    def activate(self, context: PluginContext) -> None:
        self.context = context
        
        # Build your custom PyQt6 widget
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel("Welcome to My Custom Tool"))
        
        # Mount your widget into the IDE sidebar
        context.add_sidebar_tab(
            tab_id="my_tool",
            widget=self.widget,
            title="Custom Tool",
            icon_name="terminal",
            short_label="Tool"
        )
        
        # Read or write persistent plugin-scoped data
        saved_counter = context.storage.get("click_count", 0)
        
    def deactivate(self) -> None:
        # Crucial: Unmount UI and delete widget when toggled off
        if self.context:
            self.context.remove_sidebar_tab("my_tool")
            self.widget.deleteLater()
            self.widget = None
            self.context = None</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>2. Authoring RL Method Plugins (<code>kind: "method"</code>)</h3>
<p>Method plugins contribute reinforcement learning algorithms (such as PPO, CQL, IQL, or hybrid agents). They integrate directly into the PyTorch Lightning training driver and Hydra configuration tree.</p>

<h4>Directory Structure</h4>
<pre><code>my_method/
├── __init__.py       # REQUIRED: Exposes the agent and triggers @register_agent
├── agent.py          # PyTorch Lightning module implementation
├── my_method.yaml    # Default Tier 1 hyperparameters (deployed to in/config/agent/)
└── plugin.json       # Optional component metadata</code></pre>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Crucial Rule:</b> The directory <i>must</i> contain an <code>__init__.py</code> file. Theta-IDE's agent registry uses <code>pkgutil.iter_modules()</code> to auto-discover modules in <code>src/usr/methods/</code>. Without <code>__init__.py</code>, the subdirectory will not be imported!
</div>

<h4>Python Implementation (<code>agent.py</code>)</h4>
<pre><code>import torch
from src.usr.methods.base_agent import OfflineAgentBase  # or OnlineAgentBase
from src.usr.methods.agent_registry import register_agent

@register_agent("my_cql", "my_cql_variant")
class MyCQLAgent(OfflineAgentBase):
    def __init__(self, obs_dim, n_actions, cfg=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cql_alpha = getattr(cfg, "cql_alpha", 5.0)
        # Initialize policy, critics, and loss criteria...

    def training_step(self, batch, batch_idx):
        obs, action, reward, next_obs, done = batch
        # Compute Bellman loss and conservative penalty
        loss = self.compute_loss(obs, action, reward, next_obs, done)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 3e-4))</code></pre>

<h4>Export in <code>__init__.py</code></h4>
<pre><code>from .agent import MyCQLAgent

__all__ = ["MyCQLAgent"]</code></pre>

<h4>Default Configuration (<code>my_method.yaml</code>)</h4>
<p>Deployed automatically to <code>in/config/agent/my_method.yaml</code>:</p>
<pre><code># @package _global_
agent:
  name: my_cql
  lr: 3e-4
  cql_alpha: 5.0
  batch_size: 256
  gamma: 0.99</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>3. Authoring Model Architecture Plugins (<code>kind: "model"</code>)</h3>
<p>Model plugins supply neural network architectures, encoders, or composite models. They register with <code>src.app.core.model_registry</code> and can be used standalone or as constituent sub-modules within larger composite pipelines.</p>

<h4>Directory Structure</h4>
<pre><code>my_transformer/
├── __init__.py       # Exposes model class and triggers @register_model
├── model.py          # PyTorch nn.Module implementing protocols
└── my_transformer.yaml # Default Tier 1 model config (deployed to in/config/model/)</code></pre>

<h4>Implementing Model Protocols</h4>
<p>Theta-IDE provides protocols in <code>src.app.core.protocols</code> to allow advanced models to communicate with the training pipeline without hardcoded coupling:</p>
<ul>
  <li><b><code>DynamicTopologyProtocol</code>:</b> For models that grow or prune rules/neurons during training (e.g. CEW). Tells the agent when topology changes so optimizers can rebind.</li>
  <li><b><code>ExtraStateProtocol</code>:</b> For saving/loading non-tensor states (e.g. symbolic rules, cluster prototypes) into checkpoints.</li>
  <li><b><code>HasModelCallbacks</code>:</b> For models that require dedicated PyTorch Lightning callbacks.</li>
</ul>

<h4>Python Implementation (<code>model.py</code>)</h4>
<pre><code>import torch.nn as nn
from src.app.core.model_registry import register_model
from src.app.core.protocols import DynamicTopologyProtocol, ExtraStateProtocol

@register_model("decision_transformer", "dt")
class DecisionTransformer(nn.Module, DynamicTopologyProtocol, ExtraStateProtocol):
    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self._changed = False

    def forward(self, x):
        return self.net(x)

    # DynamicTopologyProtocol
    def has_topology_changed(self) -> bool:
        return self._changed

    def reset_topology_changed(self) -> None:
        self._changed = False

    def clone_topology_to(self, target: nn.Module) -> None:
        target.load_state_dict(self.state_dict())

    # ExtraStateProtocol
    def extra_state(self) -> dict:
        return {"custom_metadata": "v1.0"}

    def load_extra_state(self, state: dict) -> None:
        pass</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>4. Authoring Environment Plugins (<code>kind: "env"</code>)</h3>
<p>Environment plugins package simulator definitions, reward shaping wrappers, and domain valuation logic.</p>

<h4>Directory Structure</h4>
<pre><code>my_custom_env/
├── __init__.py       # Registration or simulator hooks
├── env.py            # Environment wrapper or vectorization
├── reward.py         # Potential-based reward shaping functions
└── my_custom_env.yaml# Deployed to in/config/env/my_custom_env.yaml</code></pre>

<h4>Declarative Environment Configuration</h4>
<p>Every environment declares its operational properties declaratively in its YAML:</p>
<pre><code># @package _global_
env:
  name: my_custom_env
  offline_only: false          # Set true for static dataset-only environments
  monitor_metric: "eval/reward" # Target metric for early stopping and tuning
  preprocess_on_load: false
  obs_dim: 8
  n_actions: 4
  default_plots:
    - convergence
    - losses</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>5. Authoring Experiment Recipe Plugins (<code>kind: "experiment"</code>)</h3>
<p>Experiment recipes tie environments, agents, paradigms, and cluster resources into reproducible benchmarks.</p>
<pre><code># in/config/experiment/benchmark/my_experiment.yaml
# @package _global_
defaults:
  - /env: cartpole
  - /agent: ppo
  - /model: dueling_resnet

paradigm: online_rl
online_methods:
  - ppo
  - sac

resources:
  time: "02:00:00"
  gpus: 1
  cores: 8
  memory: "16G"

total_timesteps: 100000
eval_episodes: 20</code></pre>

<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;" />

<h3>6. Packaging & Publishing to the Community Hub</h3>

<h4>Creating the Release Archive</h4>
<p>Compress your component files into a <code>.zip</code> file:</p>
<pre><code>zip -r my_cql-1.0.0.zip my_method/</code></pre>

<h4>Generating the SHA-256 Checksum</h4>
<p>Theta-IDE enforces cryptographic integrity verification before extraction:</p>
<pre><code>shasum -a 256 my_cql-1.0.0.zip
# Example output: a1b2c3d4e5f6...</code></pre>

<h4>Publishing in the Hub Registry (<code>dist/index.json</code>)</h4>
<p>Add your component entry to the Hub's index:</p>
<pre><code>{
  "id": "my_cql",
  "name": "Custom Conservative Q-Learning",
  "kind": "method",
  "version": "1.0.0",
  "description": "Robust offline RL with adaptive conservatism penalties.",
  "author": { "name": "Your Name", "github": "yourhandle" },
  "tags": ["rl", "offline", "cql"],
  "releases": {
    "1.0.0": {
      "tag": "v1.0.0",
      "url": "https://github.com/yourhandle/my_cql/releases/download/v1.0.0/my_cql-1.0.0.zip",
      "sha256": "a1b2c3d4e5f6...",
      "size_bytes": 14200
    }
  }
}</code></pre>
<p>Once published, the component is immediately searchable, installable, and updatable via the <b>Community Hub</b> pane in Theta-IDE.</p>
""",
    ),
    DocArticle(
        id="cli_workflows",
        title="Command-Line (CLI) Workflows",
        category="Workflow & Tools",
        summary="CLI reference for run_pipeline.py: running experiments, overrides, plotting, sweeps, and cluster submission.",
        keywords=["cli", "terminal", "commands", "run_pipeline", "plot", "optuna", "sweeps", "overrides"],
        html_content="""
<h2>CLI & Command Reference</h2>
<p>Everything the IDE does is backed by a scriptable CLI. Run these from the repository root, in the backend's virtual environment.</p>

<h3>Running Experiments</h3>
<p>Experiments are named <code>&lt;group&gt;/&lt;experiment&gt;</code>, matching their path under <code>in/config/experiment/</code>:</p>
<pre><code># Quick CartPole smoke test
python run_pipeline.py cartpole/quick_test

# Full CartPole experiment
python run_pipeline.py cartpole/final_cartpole

# Override any config value with Hydra key=value syntax
python run_pipeline.py cartpole/final_cartpole total_timesteps=50000 eval_episodes=10

# Check that a config composes and validates, without training
python run_pipeline.py cartpole/quick_test dry_run=true</code></pre>
<p>Run <code>python run_pipeline.py --help</code> for the full list of orchestration overrides.</p>

<h3>Common Overrides</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Override</td>
    <td>Effect</td>
  </tr>
  <tr><td><code>dry_run=true</code></td><td>Validate the config and exit</td></tr>
  <tr><td><code>plot_only=true</code></td><td>Run only the plotting phase</td></tr>
  <tr><td><code>no_plot=true</code></td><td>Skip automatic plotting</td></tr>
  <tr><td><code>no_online=true</code> / <code>no_offline=true</code></td><td>Skip the online or offline training phase</td></tr>
  <tr><td><code>experiment_id=&lt;name&gt;</code></td><td>Override the results directory name</td></tr>
  <tr><td><code>site=ncshare</code> / <code>site=arc</code></td><td>Submit to a Slurm cluster instead of running locally (see <a href="cluster_slurm">Cluster Execution</a>)</td></tr>
</table>

<h3>Plotting</h3>
<pre><code># Re-run the plotting phase for an experiment
python run_pipeline.py cartpole/final_cartpole plot_only=true

# Or call the plot manager directly
python plot/manager.py cartpole/final_cartpole

# Start from an empty plot directory
python plot/manager.py cartpole/final_cartpole --wipe</code></pre>

<h3>Optuna Hyperparameter Sweeps</h3>
<p>Tuning experiments declare an Optuna sweeper in their config:</p>
<pre><code>python run_pipeline.py mimic/tune_mimic_all sweep=true

# Watch the sweep in the Optuna dashboard
python run_pipeline.py mimic/tune_mimic_all sweep=true dash=true</code></pre>
""",
    ),
    DocArticle(
        id="workflows_pane",
        title="Workflows Pane: String Diagrams",
        category="Workflow & Tools",
        summary="Compose multi-stage experiments as typed string diagrams, validate the wiring, and run them through the pipeline.",
        keywords=["workflow", "workflows", "string diagram", "dag", "nodes", "wires", "ports", "preset", "pipeline", "distillation", "transfer"],
        html_content=r"""
<h2>Workflows Pane: String Diagrams</h2>
<p>A <b>workflow</b> chains several training stages together. Each stage is a <b>node</b>, and typed <b>wires</b> (strings) carry artifacts such as datasets and checkpoints from one node's outputs to another's inputs. Open it from <a href="ide://workflows">View &rarr; Workflows</a>.</p>

<h3>Building a Diagram</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Control</td>
    <td>What it does</td>
  </tr>
  <tr><td><b>+ Add Node</b></td><td>Add a stage to the canvas</td></tr>
  <tr><td>Drag between ports</td><td>Connect an output port to an input port with a wire</td></tr>
  <tr><td><b>Validate Strings</b></td><td>Check every wire and required input (see below)</td></tr>
  <tr><td><b>Zoom Fit</b></td><td>Fit the whole diagram in view</td></tr>
  <tr><td><b>Load YAML&hellip; / Save YAML&hellip;</b></td><td>Open or save a diagram file</td></tr>
  <tr><td><b>Delete Node / Disconnect Wire</b></td><td>Remove the selected node or wire</td></tr>
  <tr><td><b>Clear</b></td><td>Empty the canvas</td></tr>
</table>

<h3>Node Types</h3>
<ul>
  <li><b>Training stages:</b> Online RL, Offline RL, and Supervised nodes. Each points at an existing experiment config.</li>
  <li><b>Data:</b> Dataset Source supplies a stored transition dataset.</li>
  <li><b>Transforms:</b> Feature Augmenter, Reward Shaper, and Distillation reshape data or models between stages.</li>
  <li><b>Evaluation:</b> Plot Evaluator turns metrics into plots.</li>
</ul>

<h3>Port Types</h3>
<p>Every port carries one artifact type, and wires only connect matching types: <code>dataset</code>, <code>checkpoint</code>, <code>value_estimator</code>, <code>metrics</code>, <code>logic_rules</code>, and <code>generic</code>. <b>Validate Strings</b> reports wires between mismatched types, wires to ports that don't exist, and required inputs that aren't connected.</p>

<h3>Built-In Presets</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Preset</td>
    <td>Pattern</td>
  </tr>
  <tr><td><code>transfer_learning</code></td><td>Train on one environment, then warm-start fine-tuning on another</td></tr>
  <tr><td><code>online_vs_offline</code></td><td>Collect data online, then train offline methods on it for comparison</td></tr>
  <tr><td><code>model_distillation</code></td><td>Distill a trained teacher into a smaller student model</td></tr>
  <tr><td><code>sepsis_reciprocal</code></td><td>Alternate offline RL and early prediction, feeding each back into the other</td></tr>
</table>
<p>Their definitions live in <code>in/config/workflow/</code>.</p>

<h3>Running a Workflow</h3>
<p>Workflows aren't launched from this pane. Save the diagram to <code>in/config/workflow/&lt;id&gt;.yaml</code>, then reference it from an experiment:</p>
<pre><code># in/config/experiment/mimic/reciprocal_refinement.yaml
workflow: sepsis_reciprocal</code></pre>
<p>Launching that experiment (<code>F5</code>, or <code>python run_pipeline.py mimic/reciprocal_refinement</code>) runs the nodes in dependency order. Nodes with no dependencies on each other run in the same stage.</p>
""",
    ),
    DocArticle(
        id="results_browser",
        title="Results Browser",
        category="Interface",
        summary="Find past runs, reload their configs, compare two runs side by side, and open their plots.",
        keywords=["results", "runs", "history", "compare", "diff", "reward", "seed", "plots", "records"],
        html_content=r"""
<h2>Results Browser</h2>
<p>The <a href="ide://results">Results</a> pane lists every run launched from Theta-IDE. It is hidden by default; turn it on in <i>Settings &rarr; Appearance</i> or open it with <code>Action + 4</code>.</p>

<h3>The Run Table</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Column</td>
    <td>Meaning</td>
  </tr>
  <tr><td><b>Experiment</b></td><td>The run's experiment ID, <code>&lt;name&gt;_&lt;YYYYmmdd-HHMMSS&gt;</code></td></tr>
  <tr><td><b>Seed</b></td><td>Random seed used for the run</td></tr>
  <tr><td><b>Status</b></td><td>The run's current status, such as running or completed</td></tr>
  <tr><td><b>Reward</b></td><td>The run's latest evaluation reward (<code>&mdash;</code> before the first evaluation)</td></tr>
  <tr><td><b>Source</b></td><td><i>Trained</i> for a real backend run, <i>Simulated</i> for a demo run</td></tr>
</table>
<p>Type in the search box to filter runs by name, seed, or status.</p>
<p>Double-click a row to open that run in the <a href="ide://monitor">Training Monitor</a>.</p>

<h3>Actions</h3>
<ul>
  <li><b>Load saved config:</b> load the exact configuration the selected run used back into the config editor, so you can rerun or tweak it.</li>
  <li><b>Compare two runs:</b> select two rows (Ctrl+click) to see a <i>Parameter / Run A / Run B</i> table of every setting that differs.</li>
  <li><b>View plot:</b> open the selected run's plots in the Plot Viewer.</li>
</ul>

<h3>Where Records Are Stored</h3>
<p>Run records are saved in <code>.thetaide/runs/</code> (change it with <code>--data-dir</code>). Training outputs such as metrics, checkpoints, and plots are under <code>results/</code>; see <a href="troubleshooting">Troubleshooting &amp; FAQ</a> for the layout.</p>
""",
    ),
    DocArticle(
        id="tensorboard_tab",
        title="TensorBoard Tab",
        category="Interface",
        summary="View TensorBoard dashboards inside the IDE for runs launched with Log to TensorBoard.",
        keywords=["tensorboard", "logs", "dashboard", "scalars", "webengine", "browser"],
        html_content=r"""
<h2>TensorBoard Tab</h2>
<p>Runs launched with <b>Log to TensorBoard</b> (on by default) also write TensorBoard logs to <code>results/tensorboard/</code>. The <a href="ide://tensorboard">TensorBoard</a> tab shows them without leaving the IDE. It is hidden by default; open it with <code>Action + 6</code> or <i>View &rarr; TensorBoard</i>.</p>

<h3>How It Works</h3>
<ul>
  <li>The first time you open the tab, the backend starts a local TensorBoard server, bound to <code>127.0.0.1</code> on a free port, and the tab embeds it.</li>
  <li>The server keeps running while the backend runs, so later visits open instantly.</li>
  <li><b>Stop</b> shuts the server down. <b>Open in browser</b> shows the same dashboard in your web browser.</li>
</ul>

<h3>Requirements</h3>
<ul>
  <li>The backend must be running. TensorBoard is started by the backend, not by the frontend.</li>
  <li>The embedded view needs <code>PyQt6-WebEngine</code>, which <code>frontend/requirements.txt</code> installs. Without it, the tab only offers <b>Open in browser</b>.</li>
</ul>

<div style="background-color: rgba(254, 128, 25, 0.12); border-left: 4px solid #fe8019; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <b>Stop the backend with Ctrl+C.</b> That also shuts TensorBoard down. Force-killing the backend leaves the TensorBoard server running; see <a href="troubleshooting">Troubleshooting &amp; FAQ</a> to stop it.
</div>
""",
    ),
    DocArticle(
        id="console_pane",
        title="Console",
        category="Interface",
        summary="The Console streams pipeline output and IDE messages, and accepts a few quick commands.",
        keywords=["console", "log", "output", "commands", "status", "help"],
        html_content=r"""
<h2>Console</h2>
<p>The <a href="ide://console">Console</a> pane is the IDE's log. It streams the training pipeline's output while a run is active, along with IDE messages such as saves, queue changes, and connection status.</p>
<p>It is not a shell. For an interactive shell, use the <a href="ide://terminal">Terminal</a> tab (macOS and Linux only).</p>

<h3>Commands</h3>
<p>Type a command into the input line at the bottom and press Enter:</p>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Command</td>
    <td>What it does</td>
  </tr>
  <tr><td><code>help</code></td><td>List the available commands</td></tr>
  <tr><td><code>status</code></td><td>Show the active run: its status, experiment ID, and backend job ID</td></tr>
  <tr><td><code>config</code></td><td>Print the <code>run_pipeline.py</code> command equivalent to the loaded config, to run it outside the IDE</td></tr>
  <tr><td><code>clear</code></td><td>Clear the console</td></tr>
</table>
""",
    ),
    DocArticle(
        id="settings_guide",
        title="Settings & settings.toml",
        category="Customization",
        summary="Tour of the six Settings pages, and how settings.toml files store and override your preferences.",
        keywords=["settings", "preferences", "settings.toml", "toml", "backend", "storage", "plugins", "hotkeys", "config"],
        html_content=r"""
<h2>Settings & settings.toml</h2>
<p>Open Settings with <code>Action + 0</code> or <i>View &rarr; Settings &amp; About</i>.</p>

<h3>Settings Pages</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Page</td>
    <td>What you can change</td>
  </tr>
  <tr><td><b>Appearance</b></td><td>Color theme, which sidebar panels are shown, auto-hide sidebar, and the sculpture animation</td></tr>
  <tr><td><b>Core Plugins</b></td><td>Turn built-in plugins (such as this Documentation plugin) on or off</td></tr>
  <tr><td><b>Community Plugins</b></td><td>Install, configure, or remove plugins from the Community Hub</td></tr>
  <tr><td><b>Hotkeys</b></td><td>Action key, terminal precedence, the <code>Action + digit</code> pane mapping, and menu shortcuts</td></tr>
  <tr><td><b>Backend API</b></td><td>Backend URL and connection status</td></tr>
  <tr><td><b>Workspace &amp; Storage</b></td><td>Data and database locations, open the settings file, and reset the UI layout</td></tr>
</table>

<h3>Settings Files</h3>
<p>Every setting is stored in a TOML file you can also edit by hand. Two files are read, and the workspace file wins:</p>
<ol>
  <li><code>~/.config/thetaide/settings.toml</code>: your defaults for every workspace (on Windows, <code>C:\Users\&lt;you&gt;\.config\thetaide\settings.toml</code>).</li>
  <li><code>.thetaide/settings.toml</code> in the repository: this workspace. Created with defaults on first launch.</li>
</ol>
<p>Open the workspace file with <i>View &rarr; Preferences: Open Settings File</i>. Changes you save in an editor apply immediately, without restarting.</p>

<h3>Sections</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Section</td>
    <td>Keys</td>
  </tr>
  <tr><td><code>[appearance]</code></td><td><code>theme</code>, plus <code>ascii_*</code> animation settings</td></tr>
  <tr><td><code>[backend]</code></td><td><code>url</code>, <code>timeout</code> (seconds)</td></tr>
  <tr><td><code>[sidebar]</code></td><td><code>order</code> and <code>visible</code> pane lists</td></tr>
  <tr><td><code>[plugins]</code></td><td><code>enabled</code> plugin IDs</td></tr>
  <tr><td><code>[hotkeys]</code></td><td><code>enabled</code>, <code>action_key</code>, <code>leader_timeout</code>, <code>terminal_precedence</code>, <code>panes</code></td></tr>
  <tr><td><code>[shortcuts]</code></td><td>Menu shortcut overrides, e.g. <code>launch_training = "F6"</code></td></tr>
</table>

<p>See also <a href="themes">Themes &amp; Theme Builder</a>, <a href="sidebar_layout">Sidebar &amp; Layout</a>, and <a href="hotkeys">Keyboard Shortcuts</a>.</p>
""",
    ),
    DocArticle(
        id="themes",
        title="Themes & Theme Builder",
        category="Customization",
        summary="Switch between the built-in color themes or design your own with the Theme Builder.",
        keywords=["theme", "themes", "colors", "palette", "dark", "light", "gruvbox", "catppuccin", "nord", "dracula", "theme builder"],
        html_content=r"""
<h2>Themes & Theme Builder</h2>

<h3>Switching Themes</h3>
<p>Pick a theme from <i>View &rarr; Themes</i>, or from <i>Settings &rarr; Appearance &rarr; Active Theme</i>. The change applies immediately and is saved as <code>theme</code> under <code>[appearance]</code> in <code>settings.toml</code>.</p>

<h3>Built-In Themes</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>Family</td>
    <td>Themes</td>
  </tr>
  <tr><td>Catppuccin</td><td>Catppuccin (default), Catppuccin Macchiato, Catppuccin Latte</td></tr>
  <tr><td>Gruvbox</td><td>Gruvbox Dark, Gruvbox Light</td></tr>
  <tr><td>Classic</td><td>Nord, Dracula, Paper</td></tr>
  <tr><td>Olympian</td><td>Apollo, Athena, Ares, Dionysus, Poseidon</td></tr>
</table>

<h3>Theme Builder</h3>
<p>Open <i>View &rarr; Theme builder&hellip;</i> to create your own palette:</p>
<ol>
  <li>Under <b>Start from</b>, pick the preset to base your theme on.</li>
  <li>Give it a <b>Theme name</b>.</li>
  <li>Click <b>Choose&hellip;</b> next to any color role to change it. Valid edits preview across the whole app as you make them.</li>
  <li>Click <b>Save &amp; apply</b>. Your theme then appears in the Themes menu alongside the built-in ones.</li>
</ol>
<p><b>Reset to preset</b> discards your edits, and <b>Cancel</b> restores the theme you had before opening the builder.</p>
""",
    ),
    DocArticle(
        id="sidebar_layout",
        title="Sidebar & Layout",
        category="Customization",
        summary="Show or hide panels, reorder sidebar tabs, auto-hide the sidebar, and restore the default layout.",
        keywords=["sidebar", "layout", "panels", "tabs", "hide", "show", "reorder", "auto-hide", "reset", "restore"],
        html_content=r"""
<h2>Sidebar & Layout</h2>

<h3>Showing and Hiding Panels</h3>
<p>To keep the sidebar short, four panels are <b>hidden by default</b>: Results, Plots, TensorBoard, and Queue. Turn any panel on or off with its switch in <i>Settings &rarr; Appearance &rarr; Sidebar Panels Visibility</i>. At least one panel must stay visible, so the last switch can't be turned off.</p>
<p>A hidden panel is still reachable: choose it from the <i>View</i> menu, or press its <code>Action + digit</code> hotkey.</p>

<h3>Reordering Tabs</h3>
<p>Drag a sidebar tab up or down to move it. The new order is saved as <code>order</code> under <code>[sidebar]</code> in <code>settings.toml</code>.</p>

<h3>Auto-Hide Sidebar</h3>
<p>Turn on <i>View &rarr; Auto-hide sidebar</i> (or the switch in <i>Settings &rarr; Appearance</i>) to give the panels the full window width. Hover over the left edge of the window to bring the sidebar back.</p>

<h3>Restoring Defaults</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; border-color: rgba(255,255,255,0.15);">
  <tr style="background-color: rgba(255,255,255,0.05); font-weight: bold;">
    <td>To reset</td>
    <td>Use</td>
  </tr>
  <tr><td>Tab order and visibility</td><td><i>Settings &rarr; Appearance &rarr; Restore default sidebar</i> (shows all panels)</td></tr>
  <tr><td>Pane sizes and docking</td><td><i>View &rarr; Restore default layout</i>, or <i>Settings &rarr; Workspace &amp; Storage &rarr; Reset UI layout</i></td></tr>
</table>
""",
    ),
    DocArticle(
        id="troubleshooting",
        title="Troubleshooting & FAQ",
        category="Getting Started",
        summary="Answers to common setup, backend, OpenGL, and cluster connection questions.",
        keywords=["troubleshooting", "faq", "errors", "backend", "connection", "opengl", "debug"],
        html_content="""
<h2>Troubleshooting & FAQ</h2>

<h3>1. Backend status says "connecting…" or "offline"</h3>
<p>Theta-IDE communicates with a background FastAPI daemon (default port: <code>8000</code>). If the backend is not responding:</p>
<ul>
  <li>Check <i>Settings &rarr; Backend API</i> to verify the daemon URL (typically <code>http://127.0.0.1:8000</code>).</li>
  <li>Ensure no other application is holding port 8000.</li>
  <li>Simulated demo runs (<i>Run &rarr; Start simulated demo</i>) do not require a live backend.</li>
</ul>

<h3>2. OpenGL / Qt Display Errors on Headless or Cluster Nodes</h3>
<p>When running tests or GUI components on a headless server, export the offscreen platform plugin:</p>
<pre><code>export QT_QPA_PLATFORM=offscreen</code></pre>

<h3>3. Where are my training results and logs stored?</h3>
<p>Results adhere to a standardized hierarchical structure:</p>
<ul>
  <li><b>Metrics & Logs:</b> <code>results/logs/[GROUP]/[EXP_ID]/[AGENT]/version_X/metrics.csv</code></li>
  <li><b>Checkpoints:</b> <code>results/checkpoints/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Replay Buffers:</b> <code>results/datasets/[GROUP]/[EXP_ID]/[AGENT]/</code></li>
  <li><b>Plots & Reports:</b> <code>results/plots/[GROUP]/[EXP_ID]/</code></li>
</ul>

<h3>4. How do I reset the IDE's layout or cached settings?</h3>
<ul>
  <li><b>Sidebar tabs:</b> <i>Settings &rarr; Appearance &rarr; Restore default sidebar</i> shows all panels and restores the original tab order.</li>
  <li><b>Pane sizes and docking:</b> <i>Settings &rarr; Workspace &amp; Storage &rarr; Reset UI layout</i>, or <i>View &rarr; Restore default layout</i>.</li>
  <li><b>Everything else:</b> edit or delete <code>.thetaide/settings.toml</code>. Theta-IDE recreates it with defaults. See <a href="settings_guide">Settings &amp; settings.toml</a>.</li>
</ul>

<h3>5. Windows: <code>Activate.ps1 cannot be loaded because running scripts is disabled</code></h3>
<p>PowerShell blocks the virtual environment's activation script by default. Allow it once for your account:</p>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser RemoteSigned</code></pre>

<h3>6. Windows: <code>Python was not found; run without arguments to install from the Microsoft Store</code></h3>
<p>Windows is using its Store shortcut instead of your Python install. Turn off the <code>python.exe</code> and <code>python3.exe</code> entries in <i>Settings &rarr; Apps &rarr; Advanced app settings &rarr; App execution aliases</i>, or activate the virtual environment first.</p>

<h3>7. The Terminal tab doesn't work on Windows</h3>
<p>The embedded terminal needs a POSIX pseudo-terminal, which Windows doesn't provide. Use a separate PowerShell window instead.</p>

<h3>8. Linux: <code>libEGL.so.1: cannot open shared object file</code></h3>
<p>Qt needs system graphics libraries that minimal Linux installs and CI runners often lack:</p>
<pre><code>sudo apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3</code></pre>

<h3>9. TensorBoard keeps running after the backend was force-closed</h3>
<p>Stop the backend with <code>Ctrl+C</code> so it shuts TensorBoard down too. If the backend was killed another way, find the leftover process. It runs as <code>python -m tensorboard.main</code>:</p>
<pre><code># macOS / Linux
pkill -f tensorboard.main

# Windows PowerShell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object CommandLine -like "*tensorboard.main*" |
    ForEach-Object { Stop-Process -Id $_.ProcessId }</code></pre>
""",
    ),
]


def get_all_articles() -> List[DocArticle]:
    """Return all available documentation articles."""
    return list(ARTICLES)


def get_article_by_id(article_id: str) -> DocArticle | None:
    """Retrieve an article by its unique identifier."""
    for art in ARTICLES:
        if art.id == article_id:
            return art
    return None


def search_articles(query: str) -> List[DocArticle]:
    """Filter articles by search query against title, summary, keywords, and content."""
    query = query.strip().lower()
    if not query:
        return get_all_articles()

    results: List[DocArticle] = []
    for art in ARTICLES:
        if (
            query in art.title.lower()
            or query in art.summary.lower()
            or query in art.category.lower()
            or any(query in kw.lower() for kw in art.keywords)
            or query in art.html_content.lower()
        ):
            results.append(art)
    return results

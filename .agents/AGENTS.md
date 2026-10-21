# Theta-IDE Workspace Guidance

Theta-IDE is a general-purpose research IDE and experiment pipeline. Keep the core focused on configuration, orchestration, user interface, shared protocols, and plugin discovery. Research algorithms, models, environments, datasets, runners, and domain-specific hooks belong to installable components unless they are generic infrastructure.

## Development practices

- Use `python run_pipeline.py <experiment>` as the experiment entry point and keep configuration in the existing config system.
- Do not add one-off scripts to the repository root or `scripts/`; put temporary debugging code in the local scratch area.
- Integrate components through the public registries, protocols, and plugin entry points. Do not add core branches keyed to a particular plugin, algorithm, model, or environment.
- A plugin should package and declare the runner, environment, hooks, dependencies, and configuration it needs. The IDE should discover and invoke those through generic interfaces.
- Keep built-in paradigm, workflow, and site templates generic and usable without research plugins installed.
- Preserve existing user-authored files and avoid unrelated refactors.
- Prefer focused changes and explain any compatibility implications.

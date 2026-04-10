# lmscan VS Code Extension

A VS Code extension that integrates lmscan directly into your editor.

## Features

- **Inline highlighting**: AI-generated paragraphs highlighted by probability (green/yellow/red)
- **Status bar**: Overall document AI probability score
- **Right-click menu**: "Scan with lmscan" context menu option
- **Diagnostics**: AI detection warnings in the Problems panel

## Requirements

- Python 3.9+ with `lmscan` installed (`pip install lmscan`)
- VS Code 1.80+

## Installation

1. Install lmscan: `pip install lmscan`
2. Install this extension from the VS Code Marketplace (or VSIX)

## How It Works

The extension calls the `lmscan` CLI with `--format json` under the hood and renders results as:
- **Decorations**: Color-coded paragraph backgrounds
- **Diagnostics**: Warning/info markers per paragraph
- **Status bar item**: Live document score

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `lmscan.pythonPath` | string | `"python"` | Path to Python interpreter with lmscan installed |
| `lmscan.threshold` | number | `0.65` | Minimum probability to flag as AI-generated |
| `lmscan.autoScan` | boolean | `false` | Automatically scan on file save |
| `lmscan.languages` | array | `["plaintext","markdown"]` | File types to scan |

## Development

```bash
cd vscode-extension
npm install
npm run compile
# Press F5 to launch Extension Development Host
```

## Publishing

```bash
npm install -g @vscode/vsce
vsce package
vsce publish
```

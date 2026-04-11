import * as vscode from "vscode";
import { spawn } from "child_process";

// ── Decoration types ─────────────────────────────────────────────────────────

const highAiDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: "rgba(255, 60, 60, 0.15)",
  overviewRulerColor: "red",
  overviewRulerLane: vscode.OverviewRulerLane.Right,
});

const mediumAiDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: "rgba(255, 200, 0, 0.12)",
  overviewRulerColor: "yellow",
  overviewRulerLane: vscode.OverviewRulerLane.Right,
});

const lowAiDecoration = vscode.window.createTextEditorDecorationType({
  backgroundColor: "rgba(60, 200, 60, 0.08)",
});

const sentenceHighDecoration = vscode.window.createTextEditorDecorationType({
  isWholeLine: true,
  overviewRulerColor: "rgba(255, 60, 60, 0.6)",
  overviewRulerLane: vscode.OverviewRulerLane.Left,
  after: {
    contentText: " ● AI",
    color: "rgba(255, 80, 80, 0.8)",
    fontStyle: "italic",
    margin: "0 0 0 1em",
  },
});

const sentenceMedDecoration = vscode.window.createTextEditorDecorationType({
  isWholeLine: true,
  after: {
    contentText: " ◐ mixed",
    color: "rgba(255, 200, 0, 0.7)",
    fontStyle: "italic",
    margin: "0 0 0 1em",
  },
});

// ── Interfaces ───────────────────────────────────────────────────────────────

interface SentenceScore {
  ai_probability: number;
  features: Record<string, number>;
  flags: string[];
}

interface ScanResult {
  ai_probability: number;
  verdict: string;
  confidence: string;
  model_attribution: Array<{
    model: string;
    confidence: number;
    evidence: string[];
    marker_count: number;
  }>;
  features: Record<string, number>;
  sentence_scores: SentenceScore[];
  flags: string[];
  scan_time_s: number;
}

// ── State ────────────────────────────────────────────────────────────────────

let statusBarItem: vscode.StatusBarItem;
let diagnosticCollection: vscode.DiagnosticCollection;
let outputChannel: vscode.OutputChannel;

// ── Activation ───────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("lmscan");
  context.subscriptions.push(outputChannel);

  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.command = "lmscan.scanDocument";
  statusBarItem.text = "$(search) lmscan";
  statusBarItem.tooltip = "Click to scan with lmscan";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  diagnosticCollection = vscode.languages.createDiagnosticCollection("lmscan");
  context.subscriptions.push(diagnosticCollection);

  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.scanDocument", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      await scanText(editor, editor.document.getText(), false);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.scanSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const sel = editor.document.getText(editor.selection);
      if (!sel.trim()) {
        vscode.window.showWarningMessage("No text selected");
        return;
      }
      await scanText(editor, sel, false);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.deepScan", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      await scanText(editor, editor.document.getText(), true);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.clear", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      editor.setDecorations(highAiDecoration, []);
      editor.setDecorations(mediumAiDecoration, []);
      editor.setDecorations(lowAiDecoration, []);
      editor.setDecorations(sentenceHighDecoration, []);
      editor.setDecorations(sentenceMedDecoration, []);
      diagnosticCollection.clear();
      statusBarItem.text = "$(search) lmscan";
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (doc) => {
      const config = vscode.workspace.getConfiguration("lmscan");
      if (!config.get<boolean>("autoScan")) return;
      const langs = config.get<string[]>("languages") || [];
      if (!langs.includes(doc.languageId)) return;
      const ed = vscode.window.visibleTextEditors.find(
        (e) => e.document === doc
      );
      if (ed) {
        await scanText(ed, doc.getText(), false);
      }
    })
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async function runLmscan(
  pythonPath: string,
  args: string[],
  input: string
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const proc = spawn(pythonPath, ["-m", "lmscan", ...args], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let out = "";
    let err = "";
    proc.stdout.on("data", (chunk: Buffer) => {
      out += chunk.toString();
    });
    proc.stderr.on("data", (chunk: Buffer) => {
      err += chunk.toString();
    });
    proc.on("close", (code) => {
      if (
        code === 0 ||
        out.trim().startsWith("{") ||
        out.trim().startsWith("[")
      ) {
        resolve(out);
      } else {
        reject(new Error(err || `lmscan exited with code ${code}`));
      }
    });
    proc.on("error", reject);
    proc.stdin.write(input.substring(0, 200000));
    proc.stdin.end();
  });
}

// ── Main scan ────────────────────────────────────────────────────────────────

async function scanText(
  editor: vscode.TextEditor,
  text: string,
  deep: boolean
): Promise<void> {
  const config = vscode.workspace.getConfiguration("lmscan");
  const pythonPath = config.get<string>("pythonPath") || "python";
  const threshold = config.get<number>("threshold") || 0.65;

  statusBarItem.text = "$(loading~spin) Scanning...";

  try {
    const stdout = await runLmscan(pythonPath, ["--format", "json", "-"], text);
    const result: ScanResult = JSON.parse(stdout);
    const prob = result.ai_probability;
    const pct = Math.round(prob * 100);

    // Status bar
    const icon =
      prob >= 0.65 ? "$(warning)" : prob >= 0.4 ? "$(info)" : "$(check)";
    statusBarItem.text = `${icon} AI: ${pct}%`;

    const topModel = result.model_attribution.length
      ? result.model_attribution[0]
      : null;
    statusBarItem.tooltip = [
      `${result.verdict} (${result.confidence} confidence)`,
      topModel
        ? `Likely: ${topModel.model} (${Math.round(topModel.confidence * 100)}%)`
        : "",
      `${result.scan_time_s}s`,
    ]
      .filter(Boolean)
      .join("\n");

    // Output channel — full report
    outputChannel.clear();
    outputChannel.appendLine("lmscan v0.6.0 — AI Text Forensics");
    outputChannel.appendLine("═".repeat(50));
    outputChannel.appendLine(`Verdict: ${result.verdict} (${pct}% AI)`);
    outputChannel.appendLine(`Confidence: ${result.confidence}`);
    outputChannel.appendLine(
      `Words: ${result.features["word_count"] || 0}`
    );
    outputChannel.appendLine(`Scan time: ${result.scan_time_s}s`);
    outputChannel.appendLine("");

    if (topModel) {
      outputChannel.appendLine("── Model Attribution ──");
      for (const m of result.model_attribution.slice(0, 5)) {
        const star = m === topModel ? "★" : " ";
        outputChannel.appendLine(
          `  ${star} ${m.model}: ${Math.round(m.confidence * 100)}% (${m.marker_count} markers)`
        );
        if (m.evidence.length > 0) {
          outputChannel.appendLine(
            `    Evidence: ${m.evidence.join(", ")}`
          );
        }
      }
      outputChannel.appendLine("");
    }

    if (result.flags.length > 0) {
      outputChannel.appendLine("── Flags ──");
      for (const f of result.flags) {
        outputChannel.appendLine(`  ⚠ ${f}`);
      }
      outputChannel.appendLine("");
    }

    if (result.sentence_scores.length > 0) {
      outputChannel.appendLine("── Sentence Analysis ──");
      for (const ss of result.sentence_scores) {
        const bar = "█"
          .repeat(Math.round(ss.ai_probability * 10))
          .padEnd(10, "░");
        outputChannel.appendLine(
          `  [${bar}] ${Math.round(ss.ai_probability * 100)}% ${ss.flags.join(", ")}`
        );
      }
    }

    // Diagnostics
    const diagnostics: vscode.Diagnostic[] = [];
    if (prob >= threshold) {
      const range = new vscode.Range(
        0,
        0,
        editor.document.lineCount - 1,
        0
      );
      const diag = new vscode.Diagnostic(
        range,
        `AI text: ${pct}% — ${result.verdict}${topModel ? ` (likely ${topModel.model})` : ""}`,
        prob >= 0.85
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Information
      );
      diag.source = "lmscan";
      diagnostics.push(diag);
    }
    diagnosticCollection.set(editor.document.uri, diagnostics);

    // Decorations
    if (deep) {
      await applyParagraphDecorations(editor, pythonPath);
    } else {
      const fullRange = new vscode.Range(
        0,
        0,
        editor.document.lineCount - 1,
        editor.document.lineAt(editor.document.lineCount - 1).text.length
      );
      editor.setDecorations(sentenceHighDecoration, []);
      editor.setDecorations(sentenceMedDecoration, []);
      if (prob >= 0.65) {
        editor.setDecorations(highAiDecoration, [fullRange]);
        editor.setDecorations(mediumAiDecoration, []);
        editor.setDecorations(lowAiDecoration, []);
      } else if (prob >= 0.4) {
        editor.setDecorations(highAiDecoration, []);
        editor.setDecorations(mediumAiDecoration, [fullRange]);
        editor.setDecorations(lowAiDecoration, []);
      } else {
        editor.setDecorations(highAiDecoration, []);
        editor.setDecorations(mediumAiDecoration, []);
        editor.setDecorations(lowAiDecoration, [fullRange]);
      }
    }

    vscode.window.showInformationMessage(
      `lmscan: ${pct}% AI${topModel ? ` (${topModel.model})` : ""} — ${result.verdict}`
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    statusBarItem.text = "$(error) lmscan";
    if (msg.includes("ENOENT") || msg.includes("not found")) {
      vscode.window.showErrorMessage(
        "lmscan not found. Install with: pip install lmscan"
      );
    } else {
      vscode.window.showErrorMessage(`lmscan error: ${msg}`);
    }
  }
}

// ── Paragraph-level deep scan ────────────────────────────────────────────────

async function applyParagraphDecorations(
  editor: vscode.TextEditor,
  pythonPath: string
): Promise<void> {
  const doc = editor.document;
  const lines = doc.getText().split("\n");

  // Find paragraph boundaries
  const paragraphs: Array<{
    text: string;
    startLine: number;
    endLine: number;
  }> = [];
  let paraStart = -1;
  let paraLines: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim().length > 0) {
      if (paraStart < 0) paraStart = i;
      paraLines.push(lines[i]);
    } else if (paraStart >= 0) {
      paragraphs.push({
        text: paraLines.join("\n"),
        startLine: paraStart,
        endLine: i - 1,
      });
      paraStart = -1;
      paraLines = [];
    }
  }
  if (paraStart >= 0) {
    paragraphs.push({
      text: paraLines.join("\n"),
      startLine: paraStart,
      endLine: lines.length - 1,
    });
  }

  const highRanges: vscode.DecorationOptions[] = [];
  const medRanges: vscode.DecorationOptions[] = [];
  const lowRanges: vscode.DecorationOptions[] = [];
  const sentHighRanges: vscode.DecorationOptions[] = [];
  const sentMedRanges: vscode.DecorationOptions[] = [];

  for (const para of paragraphs) {
    if (para.text.trim().split(/\s+/).length < 5) continue;

    try {
      const stdout = await runLmscan(
        pythonPath,
        ["--format", "json", "-"],
        para.text
      );
      const result: ScanResult = JSON.parse(stdout);
      const prob = result.ai_probability;
      const pct = Math.round(prob * 100);
      const topModel = result.model_attribution.length
        ? result.model_attribution[0].model
        : "unknown";

      const range = new vscode.Range(
        para.startLine,
        0,
        para.endLine,
        lines[para.endLine].length
      );

      const hoverMsg = new vscode.MarkdownString(
        `**lmscan** — ${pct}% AI\n\n` +
          `Verdict: ${result.verdict}\n\n` +
          `Model: ${topModel}\n\n` +
          (result.flags.length > 0
            ? `Flags:\n${result.flags.map((f) => `- ${f}`).join("\n")}`
            : "No flags triggered")
      );

      const opt: vscode.DecorationOptions = {
        range,
        hoverMessage: hoverMsg,
      };

      if (prob >= 0.65) {
        highRanges.push(opt);
      } else if (prob >= 0.4) {
        medRanges.push(opt);
      } else {
        lowRanges.push(opt);
      }

      if (prob >= 0.65) {
        sentHighRanges.push({
          range: new vscode.Range(para.startLine, 0, para.startLine, 0),
        });
      } else if (prob >= 0.45) {
        sentMedRanges.push({
          range: new vscode.Range(para.startLine, 0, para.startLine, 0),
        });
      }
    } catch {
      // Skip failed paragraphs
    }
  }

  editor.setDecorations(highAiDecoration, highRanges);
  editor.setDecorations(mediumAiDecoration, medRanges);
  editor.setDecorations(lowAiDecoration, lowRanges);
  editor.setDecorations(sentenceHighDecoration, sentHighRanges);
  editor.setDecorations(sentenceMedDecoration, sentMedRanges);
}

export function deactivate(): void {
  diagnosticCollection?.dispose();
}

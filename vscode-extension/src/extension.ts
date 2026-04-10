import * as vscode from "vscode";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

// Decoration types for AI probability ranges
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

interface ScanResult {
  ai_probability: number;
  verdict: string;
  confidence: string;
  model_attribution: Array<{ model: string; confidence: number }>;
  features: Record<string, number>;
}

let statusBarItem: vscode.StatusBarItem;
let diagnosticCollection: vscode.DiagnosticCollection;

export function activate(context: vscode.ExtensionContext): void {
  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.command = "lmscan.scanDocument";
  statusBarItem.text = "$(search) lmscan";
  statusBarItem.tooltip = "Click to scan with lmscan";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // Diagnostics
  diagnosticCollection =
    vscode.languages.createDiagnosticCollection("lmscan");
  context.subscriptions.push(diagnosticCollection);

  // Commands
  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.scanDocument", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      await scanText(editor, editor.document.getText());
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmscan.scanSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const selection = editor.document.getText(editor.selection);
      if (!selection.trim()) {
        vscode.window.showWarningMessage("No text selected");
        return;
      }
      await scanText(editor, selection);
    })
  );

  // Auto-scan on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (doc) => {
      const config = vscode.workspace.getConfiguration("lmscan");
      if (!config.get<boolean>("autoScan")) return;
      const langs = config.get<string[]>("languages") || [];
      if (!langs.includes(doc.languageId)) return;
      const editor = vscode.window.visibleTextEditors.find(
        (e) => e.document === doc
      );
      if (editor) {
        await scanText(editor, doc.getText());
      }
    })
  );
}

async function scanText(
  editor: vscode.TextEditor,
  text: string
): Promise<void> {
  const config = vscode.workspace.getConfiguration("lmscan");
  const pythonPath = config.get<string>("pythonPath") || "python";
  const threshold = config.get<number>("threshold") || 0.65;

  statusBarItem.text = "$(loading~spin) Scanning...";

  try {
    const { stdout } = await execFileAsync(pythonPath, [
      "-m",
      "lmscan",
      "--format",
      "json",
      text.substring(0, 50000), // limit text length
    ]);

    const result: ScanResult = JSON.parse(stdout);
    const prob = result.ai_probability;
    const pct = Math.round(prob * 100);

    // Update status bar
    const icon = prob >= 0.65 ? "$(warning)" : prob >= 0.4 ? "$(info)" : "$(check)";
    statusBarItem.text = `${icon} AI: ${pct}%`;
    statusBarItem.tooltip = `${result.verdict} (${result.confidence} confidence)${
      result.model_attribution.length
        ? `\nLikely: ${result.model_attribution[0].model}`
        : ""
    }`;

    // Add diagnostics
    const diagnostics: vscode.Diagnostic[] = [];
    if (prob >= threshold) {
      const range = new vscode.Range(0, 0, editor.document.lineCount - 1, 0);
      const diag = new vscode.Diagnostic(
        range,
        `AI-generated text detected: ${pct}% probability (${result.verdict})`,
        prob >= 0.85
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Information
      );
      diag.source = "lmscan";
      diagnostics.push(diag);
    }
    diagnosticCollection.set(editor.document.uri, diagnostics);

    // Apply decorations based on probability
    const fullRange = new vscode.Range(
      0,
      0,
      editor.document.lineCount - 1,
      editor.document.lineAt(editor.document.lineCount - 1).text.length
    );
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

    vscode.window.showInformationMessage(
      `lmscan: ${pct}% AI probability — ${result.verdict}`
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

export function deactivate(): void {
  diagnosticCollection?.dispose();
}

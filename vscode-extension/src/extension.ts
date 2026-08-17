// DevPilot AI VS Code extension.
//
// The one job this file has: figure out the open workspace folder (VS
// Code hands this to us directly -- no cwd-inference needed, unlike the
// devpilot_cli.py CLI), spawn the existing FastAPI backend with that
// folder as the child process's cwd (Entry 25's cwd-based
// WORKSPACE_ROOT default does the rest automatically), and show the
// already-built, already-tested React UI (Entry 26) in a sidebar
// webview via an iframe -- not a reimplementation of the chat UI here.
//
// Full rationale in DevPilot_AI_Implementation_Log.html Entry 27.

import * as vscode from "vscode";
import * as http from "http";
import { ChildProcess, spawn } from "child_process";
import * as path from "path";
import * as fs from "fs";

const PORT = 8001;
const HEALTH_URL = `http://127.0.0.1:${PORT}/health`;
const WORKSPACE_URL = `http://127.0.0.1:${PORT}/workspace`;
const HEALTH_POLL_INTERVAL_MS = 500;
const HEALTH_POLL_TIMEOUT_MS = 60_000;
const PROCESS_EXIT_TIMEOUT_MS = 5_000;

// Only set when *this* extension host spawned the running backend --
// distinguishes "mine, safe to kill and respawn" from "someone else's
// (another window, or a manually-run `devpilot`), don't touch it."
let backendProcess: ChildProcess | null = null;
let backendProcessFolder: string | null = null;

function normalizePath(p: string): string {
  return path.resolve(p).toLowerCase();
}

function checkHealth(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: 1000 }, (res) => {
      resolve(res.statusCode === 200);
      res.resume();
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

function fetchWorkspaceRoot(): Promise<string | null> {
  return new Promise((resolve) => {
    const req = http.get(WORKSPACE_URL, { timeout: 1000 }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        try {
          resolve(JSON.parse(body).workspace_root ?? null);
        } catch {
          resolve(null);
        }
      });
    });
    req.on("error", () => resolve(null));
    req.on("timeout", () => {
      req.destroy();
      resolve(null);
    });
  });
}

async function waitForHealthy(timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await checkHealth()) return true;
    await new Promise((r) => setTimeout(r, HEALTH_POLL_INTERVAL_MS));
  }
  return false;
}

async function killOwnedBackend(): Promise<void> {
  if (!backendProcess) return;
  const proc = backendProcess;
  backendProcess = null;
  backendProcessFolder = null;
  proc.kill();
  const deadline = Date.now() + PROCESS_EXIT_TIMEOUT_MS;
  while (proc.exitCode === null && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 200));
  }
}

function resolveInstallPath(context: vscode.ExtensionContext): string {
  const configured = vscode.workspace
    .getConfiguration("devpilot")
    .get<string>("installPath");
  if (configured && configured.trim()) return configured;
  // Only correct when running via `--extensionDevelopmentPath` pointed
  // directly at vscode-extension/ (the dev-host test loop) -- a REAL
  // install lives in VS Code's own extensions folder
  // (~/.vscode/extensions/...), which has none of DevPilot's actual
  // code. Returned as a last-resort guess; validateInstallPath() below
  // is what actually decides whether to trust it.
  return path.resolve(context.extensionPath, "..");
}

function validateInstallPath(installPath: string): boolean {
  return fs.existsSync(path.join(installPath, "backend", "main.py"));
}

function resolvePythonExecutable(installPath: string): string {
  const winPython = path.join(installPath, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(winPython)) return winPython;
  const posixPython = path.join(installPath, ".venv", "bin", "python");
  if (fs.existsSync(posixPython)) return posixPython;
  return "python"; // fall back to PATH if no venv found
}

type EnsureResult =
  | { status: "ok" }
  | { status: "conflict"; otherPath: string }
  | { status: "timeout" };

function spawnBackend(
  workspaceFolder: string,
  installPath: string,
  log: vscode.OutputChannel
): Promise<boolean> {
  const python = resolvePythonExecutable(installPath);
  log.appendLine(`Starting DevPilot backend: ${python} -m uvicorn --app-dir "${installPath}" backend.main:app --port ${PORT}`);
  log.appendLine(`Working directory (workspace root DevPilot will see): ${workspaceFolder}`);

  backendProcess = spawn(
    python,
    ["-m", "uvicorn", "--app-dir", installPath, "backend.main:app", "--port", String(PORT), "--host", "127.0.0.1"],
    { cwd: workspaceFolder, windowsHide: true }
  );
  backendProcessFolder = workspaceFolder;

  backendProcess.stdout?.on("data", (d) => log.append(d.toString()));
  backendProcess.stderr?.on("data", (d) => log.append(d.toString()));
  backendProcess.on("exit", (code) => {
    log.appendLine(`DevPilot backend exited (code ${code}).`);
    backendProcess = null;
    backendProcessFolder = null;
  });

  return waitForHealthy(HEALTH_POLL_TIMEOUT_MS);
}

async function ensureBackendRunning(
  workspaceFolder: string,
  installPath: string,
  log: vscode.OutputChannel
): Promise<EnsureResult> {
  if (await checkHealth()) {
    const runningRoot = await fetchWorkspaceRoot();
    if (runningRoot && normalizePath(runningRoot) === normalizePath(workspaceFolder)) {
      log.appendLine(`DevPilot backend already running on ${PORT}, correctly scoped to ${workspaceFolder} -- reusing it.`);
      return { status: "ok" };
    }

    // Something's answering on 8001, but not scoped to this folder.
    if (backendProcess && backendProcessFolder && normalizePath(backendProcessFolder) !== normalizePath(workspaceFolder)) {
      // We started it (for a different folder) -- safe to replace.
      log.appendLine(`Backend we started was scoped to ${backendProcessFolder}, switching to ${workspaceFolder} -- restarting.`);
      await killOwnedBackend();
    } else {
      // Not ours -- another window, or a manually-run devpilot. Don't
      // touch someone else's process; surface the conflict instead of
      // silently serving the wrong project's data.
      return { status: "conflict", otherPath: runningRoot ?? "(unknown)" };
    }
  }

  const healthy = await spawnBackend(workspaceFolder, installPath, log);
  return healthy ? { status: "ok" } : { status: "timeout" };
}

class DevPilotViewProvider implements vscode.WebviewViewProvider {
  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly log: vscode.OutputChannel
  ) {}

  async resolveWebviewView(webviewView: vscode.WebviewView): Promise<void> {
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.loadingHtml();

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
      webviewView.webview.html = this.errorHtml(
        "Open a folder in VS Code first -- DevPilot needs a workspace to work on."
      );
      return;
    }

    const installPath = resolveInstallPath(this.context);
    if (!validateInstallPath(installPath)) {
      webviewView.webview.html = this.errorHtml(
        `DevPilot's install path isn't set correctly (looked for backend/main.py under ` +
          `"${installPath}", didn't find it). Open Settings, search for "devpilot.installPath", ` +
          `and set it to the DevPilot project's own folder (the one containing backend/, ` +
          `mcp_servers/, .venv/), then reopen this panel.`
      );
      return;
    }

    const result = await ensureBackendRunning(workspaceFolder, installPath, this.log);

    if (result.status === "timeout") {
      webviewView.webview.html = this.errorHtml(
        `DevPilot backend didn't come up within ${HEALTH_POLL_TIMEOUT_MS / 1000}s. Check the "DevPilot AI" output channel for details.`
      );
      return;
    }

    if (result.status === "conflict") {
      webviewView.webview.html = this.errorHtml(
        `DevPilot is already running for a different project (${result.otherPath}) on port ${PORT}. ` +
          `Stop that instance first (close its window, or stop the process), then reopen this panel.`
      );
      return;
    }

    webviewView.webview.html = this.iframeHtml();
  }

  private loadingHtml(): string {
    return `<!DOCTYPE html><html><body style="font-family:sans-serif;padding:1rem;">
      <p>Starting DevPilot AI...</p></body></html>`;
  }

  private errorHtml(message: string): string {
    return `<!DOCTYPE html><html><body style="font-family:sans-serif;padding:1rem;color:#d33;">
      <p>${message}</p></body></html>`;
  }

  private iframeHtml(): string {
    return `<!DOCTYPE html><html><body style="margin:0;padding:0;">
      <iframe src="http://127.0.0.1:${PORT}/" style="width:100%;height:100vh;border:none;"></iframe>
      </body></html>`;
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel("DevPilot AI");
  context.subscriptions.push(log);

  const provider = new DevPilotViewProvider(context, log);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("devpilot.chat", provider)
  );
}

export function deactivate(): void {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

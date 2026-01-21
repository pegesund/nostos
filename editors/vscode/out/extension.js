"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
function extLog(msg) {
    const line = `${new Date().toISOString()} ${msg}\n`;
    fs.appendFileSync('/tmp/nostos_ext.log', line);
    console.log(msg);
}
let client;
// Track registered commands to avoid duplicates within this session
const registeredCommands = new Set();
// Helper to safely register a command (ignores if already exists)
function safeRegisterCommand(context, commandId, callback) {
    if (registeredCommands.has(commandId)) {
        console.log(`Command ${commandId} already registered by us, skipping`);
        return;
    }
    // Mark as registered first to prevent race conditions
    registeredCommands.add(commandId);
    try {
        const disposable = vscode_1.commands.registerCommand(commandId, callback);
        context.subscriptions.push(disposable);
        console.log(`Registered command: ${commandId}`);
    }
    catch (e) {
        console.log(`Command ${commandId} registration failed (may already exist): ${e.message}`);
        // Don't throw - just log and continue
    }
}
function activate(context) {
    try {
        fs.unlinkSync('/tmp/nostos_ext.log');
    }
    catch { }
    extLog('=== activate() called ===');
    console.log('Nostos extension is activating...');
    // Register all commands FIRST (before starting LSP, so they work even if LSP fails)
    // Register restart command
    safeRegisterCommand(context, 'nostos.restartServer', async () => {
        if (client) {
            await client.stop();
            client = undefined;
        }
        startLanguageServer(context);
        vscode_1.window.showInformationMessage('Nostos language server restarted');
    });
    // Register build cache command
    safeRegisterCommand(context, 'nostos.buildCache', async () => {
        if (client) {
            try {
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.buildCache',
                    arguments: []
                });
            }
            catch (e) {
                vscode_1.window.showErrorMessage(`Failed to build cache: ${e}`);
            }
        }
        else {
            vscode_1.window.showWarningMessage('Language server not running');
        }
    });
    // Register clear cache command
    safeRegisterCommand(context, 'nostos.clearCache', async () => {
        if (client) {
            try {
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.clearCache',
                    arguments: []
                });
            }
            catch (e) {
                vscode_1.window.showErrorMessage(`Failed to clear cache: ${e}`);
            }
        }
        else {
            vscode_1.window.showWarningMessage('Language server not running');
        }
    });
    // Register commit current file command (Ctrl+Alt+C)
    safeRegisterCommand(context, 'nostos.commit', async () => {
        if (client) {
            const editor = vscode_1.window.activeTextEditor;
            if (!editor) {
                vscode_1.window.showWarningMessage('No active editor');
                return;
            }
            // Only commit .nos files
            if (!editor.document.fileName.endsWith('.nos')) {
                vscode_1.window.showWarningMessage('Not a Nostos file');
                return;
            }
            try {
                const uri = editor.document.uri.toString();
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.commit',
                    arguments: [uri]
                });
                vscode_1.window.showInformationMessage('Committed to live system');
            }
            catch (e) {
                vscode_1.window.showErrorMessage(`Failed to commit: ${e}`);
            }
        }
        else {
            vscode_1.window.showWarningMessage('Language server not running');
        }
    });
    // Register commit all files command
    safeRegisterCommand(context, 'nostos.commitAll', async () => {
        if (client) {
            try {
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.commitAll',
                    arguments: []
                });
            }
            catch (e) {
                vscode_1.window.showErrorMessage(`Failed to commit all: ${e}`);
            }
        }
        else {
            vscode_1.window.showWarningMessage('Language server not running');
        }
    });
    // Start the language server AFTER registering commands
    startLanguageServer(context);
}
function startLanguageServer(context) {
    // Stop existing client if any
    if (client) {
        client.stop();
        client = undefined;
    }
    const serverPath = findServerPath(context);
    if (!serverPath) {
        vscode_1.window.showWarningMessage('Nostos language server (nostos-lsp) not found. ' +
            'Please install it or set nostos.serverPath in settings.');
        return;
    }
    // Check if file exists
    if (!fs.existsSync(serverPath)) {
        vscode_1.window.showErrorMessage(`LSP binary not found at: ${serverPath}`);
        return;
    }
    console.log(`Starting Nostos LSP server: ${serverPath}`);
    vscode_1.window.showInformationMessage(`Starting LSP: ${serverPath}`);
    // Server executable - let vscode-languageclient handle transport
    const serverExecutable = {
        command: serverPath,
        args: [],
        // Don't specify transport - let the client auto-detect stdio
        options: {
            env: { ...process.env },
            shell: false, // Direct execution, no shell wrapper
        },
    };
    const serverOptions = {
        run: serverExecutable,
        debug: serverExecutable,
    };
    // Client options
    const traceChannel = vscode_1.window.createOutputChannel('Nostos LSP Trace');
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'nostos' },
            { scheme: 'untitled', language: 'nostos' },
            { scheme: 'file', pattern: '**/*.nos' } // Also match by file pattern
        ],
        outputChannelName: 'Nostos Language Server',
        traceOutputChannel: traceChannel,
        synchronize: {
            // Watch .nos files in workspace
            fileEvents: vscode_1.workspace.createFileSystemWatcher('**/*.nos')
        }
    };
    // Log to trace channel
    traceChannel.appendLine('Starting LSP client...');
    // Create and start the client
    client = new node_1.LanguageClient('nostos', 'Nostos Language Server', serverOptions, clientOptions);
    // Add state change listener for debugging
    client.onDidChangeState((event) => {
        const stateNames = ['Starting', 'Stopped', 'Running'];
        const oldName = stateNames[event.oldState] || String(event.oldState);
        const newName = stateNames[event.newState] || String(event.newState);
        extLog(`STATE: ${oldName} -> ${newName}`);
        console.log(`LSP state change: ${oldName} -> ${newName}`);
        if (event.newState === 1) { // Stopped
            extLog('!!! SERVER STOPPED !!!');
        }
    });
    // Handle process close
    client.outputChannel.appendLine('Client initialized, starting server...');
    // Handle errors
    client.onTelemetry((data) => {
        console.log('LSP telemetry:', data);
    });
    extLog('Calling client.start()...');
    // Start the client (also starts the server)
    client.start().then(() => {
        extLog('client.start() resolved - CONNECTED');
        console.log('Nostos language server started successfully');
    }).catch((error) => {
        extLog(`client.start() FAILED: ${error.message || error}`);
        console.error('Failed to start Nostos language server:', error);
        client = undefined;
    });
    extLog('startLanguageServer() returning');
}
function findServerPath(context) {
    const config = vscode_1.workspace.getConfiguration('nostos');
    // 1. Check user-configured path
    const configuredPath = config.get('serverPath');
    if (configuredPath && fs.existsSync(configuredPath)) {
        return configuredPath;
    }
    // 2. Check bundled binary in extension
    const bundledPath = path.join(context.extensionPath, 'bin', 'nostos-lsp');
    if (fs.existsSync(bundledPath)) {
        return bundledPath;
    }
    // 3. Check common install locations
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const commonPaths = [
        path.join(homeDir, '.cargo', 'bin', 'nostos-lsp'),
        path.join(homeDir, '.local', 'bin', 'nostos-lsp'),
        '/usr/local/bin/nostos-lsp',
        '/usr/bin/nostos-lsp',
    ];
    for (const p of commonPaths) {
        if (fs.existsSync(p)) {
            return p;
        }
    }
    // 4. Try to find in PATH (will fail at runtime if not found)
    return 'nostos-lsp';
}
function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
//# sourceMappingURL=extension.js.map
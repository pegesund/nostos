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
let client;
function activate(context) {
    console.log('Nostos extension is activating...');
    // Start the language server
    startLanguageServer(context);
    // Register restart command
    context.subscriptions.push(vscode_1.commands.registerCommand('nostos.restartServer', async () => {
        if (client) {
            await client.stop();
        }
        startLanguageServer(context);
        vscode_1.window.showInformationMessage('Nostos language server restarted');
    }));
    // Register build cache command
    context.subscriptions.push(vscode_1.commands.registerCommand('nostos.buildCache', async () => {
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
    }));
    // Register clear cache command
    context.subscriptions.push(vscode_1.commands.registerCommand('nostos.clearCache', async () => {
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
    }));
    // Register commit current file command (Ctrl+Shift+O)
    // This commits the current file to the live compiler
    context.subscriptions.push(vscode_1.commands.registerCommand('nostos.commit', async () => {
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
    }));
    // Register commit all files command
    context.subscriptions.push(vscode_1.commands.registerCommand('nostos.commitAll', async () => {
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
    }));
}
function startLanguageServer(context) {
    const serverPath = findServerPath(context);
    if (!serverPath) {
        vscode_1.window.showWarningMessage('Nostos language server (nostos-lsp) not found. ' +
            'Please install it or set nostos.serverPath in settings.');
        return;
    }
    console.log(`Starting Nostos LSP server: ${serverPath}`);
    // Server executable
    const serverExecutable = {
        command: serverPath,
        args: [],
        options: {
            env: { ...process.env },
        },
    };
    const serverOptions = {
        run: serverExecutable,
        debug: serverExecutable,
    };
    // Client options
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nostos' }],
        outputChannelName: 'Nostos Language Server',
    };
    // Create and start the client
    client = new node_1.LanguageClient('nostos', 'Nostos Language Server', serverOptions, clientOptions);
    // Start the client (also starts the server)
    client.start();
    console.log('Nostos language server started');
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
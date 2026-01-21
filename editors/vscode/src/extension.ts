import * as path from 'path';
import * as fs from 'fs';
import { workspace, ExtensionContext, window, commands } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    Executable,
} from 'vscode-languageclient/node';

function extLog(msg: string) {
    const line = `${new Date().toISOString()} ${msg}\n`;
    fs.appendFileSync('/tmp/nostos_ext.log', line);
    console.log(msg);
}

let client: LanguageClient | undefined;

// Track registered commands to avoid duplicates within this session
const registeredCommands = new Set<string>();

// Helper to safely register a command (ignores if already exists)
function safeRegisterCommand(context: ExtensionContext, commandId: string, callback: (...args: any[]) => any) {
    if (registeredCommands.has(commandId)) {
        console.log(`Command ${commandId} already registered by us, skipping`);
        return;
    }

    // Mark as registered first to prevent race conditions
    registeredCommands.add(commandId);

    try {
        const disposable = commands.registerCommand(commandId, callback);
        context.subscriptions.push(disposable);
        console.log(`Registered command: ${commandId}`);
    } catch (e: any) {
        console.log(`Command ${commandId} registration failed (may already exist): ${e.message}`);
        // Don't throw - just log and continue
    }
}

export function activate(context: ExtensionContext) {
    try { fs.unlinkSync('/tmp/nostos_ext.log'); } catch {}
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
        window.showInformationMessage('Nostos language server restarted');
    });

    // Register build cache command
    safeRegisterCommand(context, 'nostos.buildCache', async () => {
        if (client) {
            try {
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.buildCache',
                    arguments: []
                });
            } catch (e) {
                window.showErrorMessage(`Failed to build cache: ${e}`);
            }
        } else {
            window.showWarningMessage('Language server not running');
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
            } catch (e) {
                window.showErrorMessage(`Failed to clear cache: ${e}`);
            }
        } else {
            window.showWarningMessage('Language server not running');
        }
    });

    // Register commit current file command (Ctrl+Alt+C)
    safeRegisterCommand(context, 'nostos.commit', async () => {
        if (client) {
            const editor = window.activeTextEditor;
            if (!editor) {
                window.showWarningMessage('No active editor');
                return;
            }

            // Only commit .nos files
            if (!editor.document.fileName.endsWith('.nos')) {
                window.showWarningMessage('Not a Nostos file');
                return;
            }

            try {
                const uri = editor.document.uri.toString();
                await client.sendRequest('workspace/executeCommand', {
                    command: 'nostos.commit',
                    arguments: [uri]
                });
                window.showInformationMessage('Committed to live system');
            } catch (e) {
                window.showErrorMessage(`Failed to commit: ${e}`);
            }
        } else {
            window.showWarningMessage('Language server not running');
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
            } catch (e) {
                window.showErrorMessage(`Failed to commit all: ${e}`);
            }
        } else {
            window.showWarningMessage('Language server not running');
        }
    });

    // Start the language server AFTER registering commands
    startLanguageServer(context);
}

function startLanguageServer(context: ExtensionContext) {
    // Stop existing client if any
    if (client) {
        client.stop();
        client = undefined;
    }

    const serverPath = findServerPath(context);

    if (!serverPath) {
        window.showWarningMessage(
            'Nostos language server (nostos-lsp) not found. ' +
            'Please install it or set nostos.serverPath in settings.'
        );
        return;
    }

    // Check if file exists
    if (!fs.existsSync(serverPath)) {
        window.showErrorMessage(`LSP binary not found at: ${serverPath}`);
        return;
    }

    console.log(`Starting Nostos LSP server: ${serverPath}`);
    window.showInformationMessage(`Starting LSP: ${serverPath}`);

    // Server executable - let vscode-languageclient handle transport
    const serverExecutable: Executable = {
        command: serverPath,
        args: [],
        // Don't specify transport - let the client auto-detect stdio
        options: {
            env: { ...process.env },
            shell: false,  // Direct execution, no shell wrapper
        },
    };

    const serverOptions: ServerOptions = {
        run: serverExecutable,
        debug: serverExecutable,
    };

    // Client options
    const traceChannel = window.createOutputChannel('Nostos LSP Trace');
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'nostos' },
            { scheme: 'untitled', language: 'nostos' },
            { scheme: 'file', pattern: '**/*.nos' }  // Also match by file pattern
        ],
        outputChannelName: 'Nostos Language Server',
        traceOutputChannel: traceChannel,
        synchronize: {
            // Watch .nos files in workspace
            fileEvents: workspace.createFileSystemWatcher('**/*.nos')
        }
    };

    // Log to trace channel
    traceChannel.appendLine('Starting LSP client...');

    // Create and start the client
    client = new LanguageClient(
        'nostos',
        'Nostos Language Server',
        serverOptions,
        clientOptions
    );

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
    client.onTelemetry((data: any) => {
        console.log('LSP telemetry:', data);
    });

    extLog('Calling client.start()...');

    // Start the client (also starts the server)
    client.start().then(() => {
        extLog('client.start() resolved - CONNECTED');
        console.log('Nostos language server started successfully');
    }).catch((error: any) => {
        extLog(`client.start() FAILED: ${error.message || error}`);
        console.error('Failed to start Nostos language server:', error);
        client = undefined;
    });

    extLog('startLanguageServer() returning');
}

function findServerPath(context: ExtensionContext): string | undefined {
    const config = workspace.getConfiguration('nostos');

    // 1. Check user-configured path
    const configuredPath = config.get<string>('serverPath');
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

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

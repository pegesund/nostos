import * as path from 'path';
import * as fs from 'fs';
import { workspace, ExtensionContext, window, commands } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    Executable,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: ExtensionContext) {
    console.log('Nostos extension is activating...');

    // Start the language server
    startLanguageServer(context);

    // Register restart command
    context.subscriptions.push(
        commands.registerCommand('nostos.restartServer', async () => {
            if (client) {
                await client.stop();
            }
            startLanguageServer(context);
            window.showInformationMessage('Nostos language server restarted');
        })
    );

    // Register build cache command
    context.subscriptions.push(
        commands.registerCommand('nostos.buildCache', async () => {
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
        })
    );

    // Register clear cache command
    context.subscriptions.push(
        commands.registerCommand('nostos.clearCache', async () => {
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
        })
    );
}

function startLanguageServer(context: ExtensionContext) {
    const serverPath = findServerPath(context);

    if (!serverPath) {
        window.showWarningMessage(
            'Nostos language server (nostos-lsp) not found. ' +
            'Please install it or set nostos.serverPath in settings.'
        );
        return;
    }

    console.log(`Starting Nostos LSP server: ${serverPath}`);

    // Server executable
    const serverExecutable: Executable = {
        command: serverPath,
        args: [],
        options: {
            env: { ...process.env },
        },
    };

    const serverOptions: ServerOptions = {
        run: serverExecutable,
        debug: serverExecutable,
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nostos' }],
        outputChannelName: 'Nostos Language Server',
    };

    // Create and start the client
    client = new LanguageClient(
        'nostos',
        'Nostos Language Server',
        serverOptions,
        clientOptions
    );

    // Start the client (also starts the server)
    client.start();

    console.log('Nostos language server started');
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

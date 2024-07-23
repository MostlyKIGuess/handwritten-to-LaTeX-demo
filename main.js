const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
let mainWindow;
let pythonProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: true,
            enableRemoteModule: true,
            webSecurity: false,
        },
        show: false,
    });
    mainWindow.setMenuBarVisibility(false)
    mainWindow.maximize()
    mainWindow.loadFile(path.join(__dirname, 'Frontend', 'build', 'index.html'));
    mainWindow.on('closed', function () {
        mainWindow = null
    });
    mainWindow.once('ready-to-show', () => {
        mainWindow.show()
    });
}

app.on('ready', () => {
    pythonProcess = spawn('python', ['app.py'], { cwd: 'Backend' });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python server output: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python server output: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python server process exited with code ${code}`);
    });

    createWindow();
});


app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

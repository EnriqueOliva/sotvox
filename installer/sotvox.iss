; Inno Setup script for Sotvox
; Build with: installer\build.ps1  (or ISCC.exe installer\sotvox.iss)

#define AppName "Sotvox"
#define AppVersion "1.0.0"
#define AppPublisher "EnriqueOliva"
#define AppURL "https://github.com/EnriqueOliva/sotvox"

[Setup]
AppId={{7C9E6A54-3B2D-4F1A-A8E7-1D2C3B4A5F60}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\Sotvox
DefaultGroupName=Sotvox
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#SourcePath}..
OutputBaseFilename=Sotvox-Setup
SetupIconFile={#SourcePath}..\assets\sotvox.ico
UninstallDisplayIcon={app}\assets\sotvox.ico
UninstallDisplayName={#AppName}
LicenseFile={#SourcePath}..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourcePath}..\launch.vbs"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}..\src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,*.pyo"
Source: "{#SourcePath}..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourcePath}..\sounds\*"; DestDir: "{app}\sounds"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourcePath}..\setup\*"; DestDir: "{app}\setup"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Sotvox"; Filename: "{app}\launch.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\assets\sotvox.ico"
Name: "{group}\Set Up or Repair Sotvox"; Filename: "{app}\setup\setup.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\assets\sotvox.ico"
Name: "{group}\Uninstall Sotvox"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Sotvox"; Filename: "{app}\launch.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\assets\sotvox.ico"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; \
    Parameters: "-ExecutionPolicy Bypass -NoProfile -NoExit -File ""{app}\setup\setup.ps1"""; \
    WorkingDir: "{app}"; \
    Description: "Install the Sotvox runtime now (downloads Python + dependencies, needs internet)"; \
    Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\log"
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\src\__pycache__"
Type: filesandordirs; Name: "{app}\src\ui\__pycache__"
Type: filesandordirs; Name: "{app}\src\workers\__pycache__"

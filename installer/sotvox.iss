; Inno Setup script for Sotvox
; Build with: installer\build.ps1  (freezes the app with PyInstaller, then compiles this)

#define AppName "Sotvox"
#define AppVersion "1.1.0"
#define AppPublisher "EnriqueOliva"
#define AppURL "https://github.com/EnriqueOliva/sotvox"
#define AppExeName "Sotvox.exe"

[Setup]
AppId={{7C9E6A54-3B2D-4F1A-A8E7-1D2C3B4A5F60}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\Sotvox
DefaultGroupName=Sotvox
DisableProgramGroupPage=yes
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#SourcePath}..
OutputBaseFilename=Sotvox-Setup
SetupIconFile={#SourcePath}..\assets\sotvox.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
LicenseFile={#SourcePath}..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoCompany={#AppPublisher}
MinVersion=10.0
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourcePath}dist_app\Sotvox\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}dist_app\Sotvox\*"; DestDir: "{app}"; Excludes: "{#AppExeName}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourcePath}..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Sotvox"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall Sotvox"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Sotvox"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,Sotvox}"; \
    WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

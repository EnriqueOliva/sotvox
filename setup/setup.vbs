Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("Shell.Application")
Set wshShell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
psScript = baseDir & "\setup.ps1"

If Not fso.FileExists(psScript) Then
    MsgBox "setup.ps1 not found in:" & vbCrLf & baseDir, vbCritical, "Sotvox Setup"
    WScript.Quit
End If

If Not WScript.Arguments.Named.Exists("elevated") Then
    shell.ShellExecute "wscript.exe", """" & WScript.ScriptFullName & """ /elevated:1", baseDir, "runas", 1
    WScript.Quit
End If

wshShell.CurrentDirectory = baseDir
wshShell.Run "powershell.exe -ExecutionPolicy Bypass -NoExit -File """ & psScript & """", 1, False

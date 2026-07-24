Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

pythonw = baseDir & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    answer = MsgBox("Sotvox needs to finish a one-time setup before it can run" & vbCrLf & _
        "(it installs Python and the required components, and needs internet)." & vbCrLf & vbCrLf & _
        "Run the setup now?", vbYesNo + vbExclamation, "Sotvox - setup required")
    If answer = vbYes Then
        setupVbs = baseDir & "\setup\setup.vbs"
        If fso.FileExists(setupVbs) Then
            shell.Run """" & setupVbs & """", 1, False
        Else
            MsgBox "Could not find the setup file:" & vbCrLf & setupVbs, vbCritical, "Sotvox"
        End If
    End If
    WScript.Quit
End If

env = "PATH"
nvidiaPath = baseDir & "\.venv\Lib\site-packages\nvidia\cublas\bin;" & baseDir & "\.venv\Lib\site-packages\nvidia\cudnn\bin;"
shell.Environment("Process")(env) = nvidiaPath & shell.Environment("Process")(env)

shell.Run """" & pythonw & """ """ & baseDir & "\src\main.py""", 0, False

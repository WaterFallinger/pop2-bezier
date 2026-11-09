Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "nsDialogs.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!ifndef VERSION
!define VERSION "0.9.29"
!endif
!ifndef STAGE
!error "Build with installer/build-universal.ps1; an isolated STAGE is required"
!endif
!ifndef SETUP_FILE
!error "SETUP_FILE is required"
!endif
!define PAYLOAD "${STAGE}\payload"
!define APPID "PrimitivePainter.Installation.v1"
!define VERSION_ID "PrimitivePainter.Universal.${VERSION}"
Name "Primitive Painter"
OutFile "${SETUP_FILE}"
InstallDir "$LOCALAPPDATA\Programs\PrimitivePainter"
InstallDirRegKey HKCU "Software\PrimitivePainter" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 32
AllowSkipFiles off
ManifestDPIAware true
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Primitive Painter"
VIAddVersionKey "FileDescription" "Primitive Painter 统一原生安装程序"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Primitive Painter Project"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "安装 Primitive Painter"
!define MUI_WELCOMEPAGE_TEXT "安装本地原生矢量绘画程序，无需编译或Python。$\r$\n$\r$\n统一入口会识别本机设备，仅准备适用的原生运行组件。若GPU不受支持或组件准备失败，将明确提供CPU兼容计算。$\r$\n$\r$\n模型权重不包含在安装包内，可安装后在设置中在线下载；已有模型位置、用户设置和画布窗口会保留。"
!define MUI_FINISHPAGE_RUN "$INSTDIR\PrimitivePainter.exe"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!insertmacro MUI_PAGE_WELCOME
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE ValidateDirectory
!insertmacro MUI_PAGE_DIRECTORY
Page custom ModelPage ModelPageLeave
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "English"
Var TestMode
Var LocalUpdate
Var RegistryKey
Var PreferenceKey
Var ShortcutDir
Var ConfigDir
Var ModelPath
Var ModelEdit
Var LogPath

Function SetupLog
  Exch $0
  Push $1
  FileOpen $1 "$LogPath" a
  FileSeek $1 0 END
  FileWrite $1 "$0$\r$\n"
  FileClose $1
  Pop $1
  Pop $0
FunctionEnd
Function .onInit
!ifdef SYNTAX_TEST
  SetErrorLevel 2
  Abort
!endif
  SetShellVarContext current
  SetRegView 64
  StrCpy $TestMode 0
  StrCpy $LocalUpdate 0
  StrCpy $LogPath "$TEMP\PrimitivePainter-setup-${VERSION}.log"
  StrCpy $RegistryKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrimitivePainter"
  StrCpy $PreferenceKey "Software\PrimitivePainter"
  ${GetParameters} $0
  ClearErrors
  ${GetOptions} $0 "/TESTMODE" $1
  ${IfNot} ${Errors}
    StrCpy $TestMode 1
    StrCpy $RegistryKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrimitivePainter.Test"
    StrCpy $PreferenceKey "Software\PrimitivePainter.Test"
  ${EndIf}
  ClearErrors
  ${GetOptions} $0 "/LOCALUPDATE" $1
  ${IfNot} ${Errors}
    StrCpy $LocalUpdate 1
  ${EndIf}
  ClearErrors
  ${GetOptions} $0 "/MODEL=" $ModelPath
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "需要64位Windows。" /SD IDOK
    SetErrorLevel 2
    Abort
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "需要Windows 10或更新版本。" /SD IDOK
    SetErrorLevel 2
    Abort
  ${EndIf}
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File "${PAYLOAD}\runtime\runtime-installer.exe"
  !include "${STAGE}\install-crt-runtime.nsh"
FunctionEnd

Function ValidateDirectory
  CreateDirectory "$INSTDIR"
  GetFullPathName $2 "$INSTDIR"
  ${If} $2 == ""
    SetErrorLevel 2
    Abort
  ${EndIf}
  StrCpy $INSTDIR $2
  ${GetRoot} "$INSTDIR" $0
  ${If} $INSTDIR == $0
  ${OrIf} $INSTDIR == "$WINDIR"
  ${OrIf} $INSTDIR == "$SYSDIR"
  ${OrIf} $INSTDIR == "$PROFILE"
  ${OrIf} $INSTDIR == "$LOCALAPPDATA"
    MessageBox MB_ICONSTOP "请选择独立的程序文件夹。" /SD IDOK
    SetErrorLevel 2
    Abort
  ${EndIf}
  IfFileExists "$INSTDIR\.primitive-painter-install" existing empty
  existing:
    FileOpen $0 "$INSTDIR\.primitive-painter-install" r
    FileRead $0 $1
    FileRead $0 $3
    FileClose $0
    StrCpy $3 $3 1
    ${If} $1 != "${APPID}$\r$\n"
    ${OrIf} $3 != $TestMode
      MessageBox MB_ICONSTOP "安装目录标记或测试范围不匹配，未覆盖现有文件。" /SD IDOK
      SetErrorLevel 2
      Abort
    ${EndIf}
    Goto version
  empty:
    ${If} $LocalUpdate == 1
      MessageBox MB_ICONSTOP "本地更新需要已有的有效安装目录。" /SD IDOK
      SetErrorLevel 2
      Abort
    ${EndIf}
    FindFirst $0 $1 "$INSTDIR\*"
    ${DoWhile} $1 != ""
      ${If} $1 != "."
      ${AndIf} $1 != ".."
        FindClose $0
        MessageBox MB_ICONSTOP "此文件夹不是本程序的安装目录，请选择空文件夹。" /SD IDOK
        SetErrorLevel 2
        Abort
      ${EndIf}
      FindNext $0 $1
    ${Loop}
    FindClose $0
  version:
    IfFileExists "$INSTDIR\runtime\app-${VERSION}\*" 0 running
    FileOpen $0 "$INSTDIR\runtime\app-${VERSION}\.primitive-painter-app" r
    IfErrors invalid_version
    FileRead $0 $1
    FileClose $0
    ${If} $1 != "${VERSION_ID}$\r$\n"
      Goto invalid_version
    ${EndIf}
    Goto running
  invalid_version:
    MessageBox MB_ICONSTOP "当前版本子目录缺少正确的所有权标记，请保留现场并选择其他安装目录。" /SD IDOK
    SetErrorLevel 2
    Abort
  running:
    nsExec::ExecToStack '"$PLUGINSDIR\runtime-installer.exe" --check-running --app-dir "$INSTDIR\runtime\app-${VERSION}"'
    Pop $0
    Pop $1
    ${If} $0 == 10
      MessageBox MB_ICONEXCLAMATION "当前${VERSION}版本窗口仍在运行。请先保存并关闭该版本后重试；其他旧版窗口可继续保留。" /SD IDOK
      SetErrorLevel 10
      Abort
    ${ElseIf} $0 != 0
      MessageBox MB_ICONSTOP "无法安全检查当前版本进程，未继续覆盖。$\r$\n$1" /SD IDOK
      SetErrorLevel 2
      Abort
    ${EndIf}
FunctionEnd

Function ModelPage
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "模型权重（可稍后选择）" "已有模型和设置会保留，不复制大型权重。"
  ${NSD_CreateLabel} 0 0 100% 36u "已有模型无需再次选择。可指定本地model.pt，或留空后在应用设置中在线下载官方权重。"
  Pop $0
  ${NSD_CreateText} 0 46u 78% 14u "$ModelPath"
  Pop $ModelEdit
  ${NSD_CreateButton} 81% 45u 19% 16u "浏览…"
  Pop $0
  ${NSD_OnClick} $0 BrowseModel
  nsDialogs::Show
FunctionEnd
Function BrowseModel
  nsDialogs::SelectFileDialog open "$ModelPath" "模型 (*.pt)|*.pt|所有文件 (*.*)|*.*"
  Pop $0
  ${If} $0 != ""
    ${NSD_SetText} $ModelEdit $0
  ${EndIf}
FunctionEnd
Function ModelPageLeave
  ${NSD_GetText} $ModelEdit $ModelPath
  ${If} $ModelPath != ""
    IfFileExists "$ModelPath" found
      MessageBox MB_ICONSTOP "模型文件不存在，可以留空稍后选择。" /SD IDOK
      Abort
    found:
  ${EndIf}
FunctionEnd

Section "Primitive Painter"
  Call ValidateDirectory
  ; Establish validated ownership before any controlled payload or lengthy
  ; dependency preparation, so an interrupted install can be resumed safely.
  ClearErrors
  FileOpen $0 "$INSTDIR\.primitive-painter-install" w
  ${If} ${Errors}
    SetErrorLevel 2
    Abort
  ${EndIf}
  FileWrite $0 "${APPID}$\r$\n$TestMode$\r$\n${VERSION_ID}$\r$\n"
  FileClose $0
  ${If} ${Errors}
    SetErrorLevel 2
    Abort
  ${EndIf}
  StrCpy $LogPath "$INSTDIR\install-${VERSION}.log"
  Push "INSTALL_START version=${VERSION} test=$TestMode local_update=$LocalUpdate"
  Call SetupLog
  SetOverwrite on
  SetOutPath "$INSTDIR\runtime\app-${VERSION}"
  ClearErrors
  FileOpen $0 "$INSTDIR\runtime\app-${VERSION}\.primitive-painter-app" w
  ${If} ${Errors}
    SetErrorLevel 2
    Abort
  ${EndIf}
  FileWrite $0 "${VERSION_ID}$\r$\n"
  FileClose $0
  ${If} ${Errors}
    SetErrorLevel 2
    Abort
  ${EndIf}
  File "${PAYLOAD}\runtime\app-${VERSION}\primitive-painter-cuda-optimized.exe"
  File "${PAYLOAD}\runtime\app-${VERSION}\primitive-painter-cuda-portable.exe"
  File "${PAYLOAD}\runtime\app-${VERSION}\primitive-painter-aten-hip.exe"
  File "${PAYLOAD}\runtime\app-${VERSION}\primitive-painter-cpu.exe"
  File "${PAYLOAD}\runtime\app-${VERSION}\painter_aten_hip.dll"
  !include "${STAGE}\install-crt-app.nsh"
  SetOutPath "$INSTDIR\runtime"
  File "${PAYLOAD}\runtime\runtime-installer.exe"
  File "${PAYLOAD}\runtime\aten-runtime-installer.exe"
  SetOverwrite off
  !include "${STAGE}\install-crt-runtime.nsh"
  SetOverwrite on
  SetOutPath "$INSTDIR\assets"
  File "${PAYLOAD}\assets\caption_spm_1500.model"
  SetOutPath "$INSTDIR"
  SetOverwrite off
  !include "${STAGE}\install-crt-root.nsh"
  SetOverwrite on
  File "${PAYLOAD}\native-runtime-manifest.json"
  File "${PAYLOAD}\README.txt"
  File "${PAYLOAD}\DIAGNOSTICS.txt"
  File "${PAYLOAD}\THIRD_PARTY_NOTICES.txt"
  !include "${STAGE}\install-licenses.nsh"
!if /FileExists "${PAYLOAD}\aten-performance-profile.json"
  File "${PAYLOAD}\aten-performance-profile.json"
!endif
  SetOverwrite off
  File "${PAYLOAD}\model-config.json"
  File "${PAYLOAD}\aten-runtime.json"
  SetOverwrite on
  ClearErrors
  File "${PAYLOAD}\PrimitivePainter.exe"
  IfErrors 0 +3
    SetErrorLevel 10
    Abort
  ${If} $TestMode == 1
    StrCpy $ConfigDir "$INSTDIR\qa-profile\PrimitivePainter"
    StrCpy $ShortcutDir "$INSTDIR\qa-shortcuts"
    CreateDirectory "$INSTDIR\qa-profile"
    System::Call 'kernel32::SetEnvironmentVariableW(w "LOCALAPPDATA", w "$INSTDIR\qa-profile")i.r0'
  ${Else}
    StrCpy $ConfigDir "$LOCALAPPDATA\PrimitivePainter"
    StrCpy $ShortcutDir "$SMPROGRAMS\Primitive Painter"
  ${EndIf}
  ${If} $ModelPath != ""
    nsExec::ExecToLog '"$INSTDIR\runtime\runtime-installer.exe" --set-model "$ModelPath" --tokenizer "$INSTDIR\assets\caption_spm_1500.model" --config-dir "$ConfigDir"'
    Pop $0
    ${If} $0 != 0
      DetailPrint "模型路径未保存；可以在应用设置中重新选择。"
    ${EndIf}
  ${EndIf}
  DetailPrint "正在按当前设备准备原生运行组件；未支持的设备可使用CPU兼容计算。"
  nsExec::ExecToLog '"$INSTDIR\PrimitivePainter.exe" --prepare-runtime --report "$INSTDIR\runtime-prepare-${VERSION}.json"'
  Pop $0
  Push "RUNTIME_PREPARE_EXIT=$0"
  Call SetupLog
  ${If} $0 != 0
    DetailPrint "运行组件检查未完成；详细信息已保留在安装目录及诊断日志中。"
  ${EndIf}
  FileOpen $0 "$INSTDIR\.primitive-painter-install" w
  FileWrite $0 "${APPID}$\r$\n$TestMode$\r$\n${VERSION_ID}$\r$\n"
  FileClose $0
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "$RegistryKey" "DisplayName" "Primitive Painter"
  WriteRegStr HKCU "$RegistryKey" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "$RegistryKey" "Publisher" "Primitive Painter Project"
  WriteRegStr HKCU "$RegistryKey" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "$RegistryKey" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
  WriteRegStr HKCU "$RegistryKey" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
  WriteRegDWORD HKCU "$RegistryKey" "NoModify" 1
  WriteRegDWORD HKCU "$RegistryKey" "NoRepair" 1
  WriteRegStr HKCU "$PreferenceKey" "InstallDir" "$INSTDIR"
  CreateDirectory "$ShortcutDir"
  CreateShortCut "$ShortcutDir\Primitive Painter.lnk" "$INSTDIR\PrimitivePainter.exe"
  CreateShortCut "$ShortcutDir\Primitive Painter（兼容模式）.lnk" "$INSTDIR\PrimitivePainter.exe" "--compatibility"
  CreateShortCut "$ShortcutDir\卸载 Primitive Painter.lnk" "$INSTDIR\Uninstall.exe"
  ${If} $TestMode != 1
    CreateShortCut "$DESKTOP\Primitive Painter.lnk" "$INSTDIR\PrimitivePainter.exe"
    CreateShortCut "$DESKTOP\Primitive Painter（兼容模式）.lnk" "$INSTDIR\PrimitivePainter.exe" "--compatibility"
  ${EndIf}
  Push "INSTALL_COMPLETE"
  Call SetupLog
SectionEnd

Function un.onInit
  SetShellVarContext current
  SetRegView 64
  GetFullPathName $INSTDIR "$INSTDIR"
  FileOpen $0 "$INSTDIR\.primitive-painter-install" r
  IfErrors invalid
  FileRead $0 $1
  FileRead $0 $TestMode
  FileRead $0 $2
  FileClose $0
  ${If} $1 != "${APPID}$\r$\n"
  ${OrIf} $2 != "${VERSION_ID}$\r$\n"
    Goto invalid
  ${EndIf}
  FileOpen $0 "$INSTDIR\runtime\app-${VERSION}\.primitive-painter-app" r
  IfErrors invalid
  FileRead $0 $1
  FileClose $0
  ${If} $1 != "${VERSION_ID}$\r$\n"
    Goto invalid
  ${EndIf}
  StrCpy $TestMode $TestMode 1
  ${If} $TestMode == 1
    StrCpy $RegistryKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrimitivePainter.Test"
    StrCpy $PreferenceKey "Software\PrimitivePainter.Test"
    StrCpy $ShortcutDir "$INSTDIR\qa-shortcuts"
  ${ElseIf} $TestMode == 0
    StrCpy $RegistryKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrimitivePainter"
    StrCpy $PreferenceKey "Software\PrimitivePainter"
    StrCpy $ShortcutDir "$SMPROGRAMS\Primitive Painter"
  ${Else}
    Goto invalid
  ${EndIf}
  nsExec::ExecToStack '"$INSTDIR\runtime\runtime-installer.exe" --check-running --app-dir "$INSTDIR\runtime\app-${VERSION}"'
  Pop $0
  Pop $1
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION "当前版本仍在运行或进程检查失败。未删除文件，请保存关闭当前版本后重试。" /SD IDOK
    SetErrorLevel 10
    Abort
  ${EndIf}
  Return
  invalid:
    MessageBox MB_ICONSTOP "安装所有权标记缺失或不匹配，未删除文件。" /SD IDOK
    SetErrorLevel 2
    Abort
FunctionEnd
Section "Uninstall"
  SetOutPath "$TEMP"
  ; Only the files embedded by this build are removed. Large runtimes, model
  ; paths, user settings, logs, unknown files and previous app versions remain.
  !include "${STAGE}\uninstall-owned.nsh"
  ; Keep both ownership receipts: prepared runtimes, settings and unknown user
  ; files survive uninstall, so the same directory must remain recognizable
  ; for a safe repair/reinstall without adopting an unowned nonempty folder.
  RMDir "$INSTDIR\assets"
  RMDir "$INSTDIR\runtime"
  ReadRegStr $0 HKCU "$RegistryKey" "InstallLocation"
  ${If} $0 == $INSTDIR
  ${OrIf} $TestMode == 1
    Delete "$ShortcutDir\Primitive Painter.lnk"
    Delete "$ShortcutDir\Primitive Painter（兼容模式）.lnk"
    Delete "$ShortcutDir\卸载 Primitive Painter.lnk"
    RMDir "$ShortcutDir"
    ${If} $TestMode != 1
      Delete "$DESKTOP\Primitive Painter.lnk"
      Delete "$DESKTOP\Primitive Painter（兼容模式）.lnk"
    ${EndIf}
  ${EndIf}
  ${If} $0 == $INSTDIR
    DeleteRegKey HKCU "$RegistryKey"
  ${EndIf}
  ReadRegStr $0 HKCU "$PreferenceKey" "InstallDir"
  ${If} $0 == $INSTDIR
    DeleteRegValue HKCU "$PreferenceKey" "InstallDir"
  ${EndIf}
  Delete "$INSTDIR\Uninstall.exe"
SectionEnd

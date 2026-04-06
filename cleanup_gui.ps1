# PowerShell script to clean up GUI changes
Write-Host "Cleaning up GUI changes..." -ForegroundColor Green

# Remove GUI-related files
Remove-Item -Path "gui_tkinter.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "launch_gui.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "demo_gui_features.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "demo_enhanced_ux.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "fix_enhanced_ux.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "gui_settings.json" -Force -ErrorAction SilentlyContinue

Write-Host "GUI files removed successfully!" -ForegroundColor Green
Write-Host "Run: python main.py to start the original system" -ForegroundColor White
@echo off
echo Joining split files back to original...

powershell -Command "& {
    $outputFile = 'nsight_profiles_restored.zip'
    $writer = [System.IO.File]::Create($outputFile)
    
    $partNumber = 1
    while ($true) {
        $partFile = 'nsight_part_' + $partNumber + '.zip'
        if (-not (Test-Path $partFile)) {
            break
        }
        
        Write-Host 'Processing:' $partFile
        $reader = [System.IO.File]::OpenRead($partFile)
        $reader.CopyTo($writer)
        $reader.Close()
        $partNumber++
    }
    
    $writer.Close()
    Write-Host 'Restoration complete!' $outputFile 'created.'
}"

pause
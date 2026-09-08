$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$texBin = 'C:\Users\lebat\AppData\Local\Programs\MiKTeX\miktex\bin\x64'
foreach ($venue in @('soict','rivf')) {
    Push-Location (Join-Path $paperRoot $venue)
    try {
        & (Join-Path $texBin 'pdflatex.exe') -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass1.txt
        if ($LASTEXITCODE -ne 0) { Get-Content build_pass1.txt -Tail 18; throw "$venue LaTeX failed" }
        & (Join-Path $texBin 'bibtex.exe') main > build_bibtex.txt
        if ($LASTEXITCODE -ne 0) { Get-Content build_bibtex.txt -Tail 18; throw "$venue bibliography failed" }
        & (Join-Path $texBin 'pdflatex.exe') -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass2.txt
        if ($LASTEXITCODE -ne 0) { Get-Content build_pass2.txt -Tail 18; throw "$venue LaTeX pass 2 failed" }
        & (Join-Path $texBin 'pdflatex.exe') -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build_pass3.txt
        if ($LASTEXITCODE -ne 0) { Get-Content build_pass3.txt -Tail 18; throw "$venue LaTeX pass 3 failed" }
        Select-String -Path main.log -Pattern 'Output written|Overfull|undefined|multiply defined' | ForEach-Object { "$venue : $($_.Line)" }
    } finally { Pop-Location }
}

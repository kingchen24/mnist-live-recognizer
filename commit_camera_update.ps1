$repoDir = 'S:\codex26\MNIST\mnist-live-recognizer'
Set-Location $repoDir

Write-Host '=== Files changed ==='
git status --short

Write-Host ''
Write-Host '=== Committing ==='
git add .
git commit -m 'docs: replace camera screenshot with digit 8 (98.9% confidence)

- New camera screenshot: hand-written "8" with 98.9% confidence
  (replaces previous "5" at 87.7%)
- Regenerated banner to use new screenshot
- Updated all README text references (EN + CN)
- Updated pic/README.md to reference camera8.png
- Better demonstrates system accuracy on a clean, well-recognized digit' 2>&1 | Out-String | Write-Host

Write-Host ''
Write-Host '=== Pushing ==='
git push 2>&1 | Out-String | Write-Host

Write-Host ''
Write-Host '=== Final repo state ==='
git log --oneline | Select-Object -First 5

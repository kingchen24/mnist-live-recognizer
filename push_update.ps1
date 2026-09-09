$repoDir = 'S:\codex26\MNIST\mnist-live-recognizer'
Set-Location $repoDir

Write-Host '=== Checking git status ==='
git status --short

Write-Host ''
Write-Host '=== Adding all changes ==='
git add .

Write-Host ''
Write-Host '=== Committing ==='
git commit -m 'docs: replace synthetic demos with real screenshots and recordings

- Replaced synthetic banner with real screenshots (mouse + camera)
- Added real camera screenshot (87.7% confidence on hand-written 5)
- Converted mouse mode MP4 to optimized 3MB GIF
- Removed outdated synthetic evolution GIFs
- Added pic/README.md documenting source assets
- Better visual impact for GitHub visitors' 2>&1 | Out-String | Write-Host

Write-Host ''
Write-Host '=== Pushing to GitHub ==='
git push 2>&1 | Out-String | Write-Host

Write-Host ''
Write-Host '=== Done ==='
gh repo view kingchen24/mnist-live-recognizer --json url,description 2>&1 | python -c "import json, sys; d=json.load(sys.stdin); print('URL: ' + d['url']); print('Description: ' + d['description'])"

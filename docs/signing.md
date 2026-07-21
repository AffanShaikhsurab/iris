# Shortcut Signing

Apple Shortcuts can be compiled from Cherri source on platforms where the Cherri
compiler runs. Modern iOS imports usually require a signed `.shortcut` file.

The known-good local path on Windows is to compile inside WSL2 with the Linux
Cherri release. Native Windows Cherri binaries are not published.

Latest local test:

- Environment: WSL2 Ubuntu on Windows.
- Cherri: `v2.3.0` Linux x86_64 release.
- Signed command:
  `cherri shortcuts/iris.cherri --hubsign --share=anyone --output='dist/Iris.shortcut'`.
- Result: `dist/Iris.shortcut` generated successfully.
- Header: `AEA1`.
- Current artifact aliases:
  - `dist/Iris.shortcut`
  - `dist/Iris.signed.shortcut`
  - `dist/iris.shortcut`

`AEA1` means the output is already in signed/package-formatted Shortcut form.
It is the file to import onto iPhone.

## Fastest reproducible path (local Cherri binary + build script)

If the repo already contains the Linux Cherri binary at `tmp/cherri-bin/cherri`
(v2.3.0), an agent can build + HubSign in one command from PowerShell in the
repo root — the build script normalizes CRLF and injects `.env.local` keys into
a throwaway copy for a personal build:

```powershell
wsl -- bash -lc "cd '/mnt/c/Users/affan/Fun Projects/siri' && chmod +x ./tmp/cherri-bin/cherri && CHERRI_BIN=./tmp/cherri-bin/cherri bash scripts/build-shortcuts.sh --hubsign --share=anyone"
```

Then mirror the conventional alias and verify the `AEA1` header:

```powershell
Copy-Item -LiteralPath "dist\Iris.shortcut" -Destination "dist\Iris.signed.shortcut" -Force
```

A `.env.local` build bakes in real keys and is **for your own device only** —
never distribute it. For a placeholder (distributable) build, temporarily move
`.env.local` aside first. Validate structure (no sign) with:

```powershell
wsl -- bash -lc "cd '/mnt/c/Users/affan/Fun Projects/siri' && CHERRI_BIN=./tmp/cherri-bin/cherri bash scripts/build-shortcuts.sh --skip-sign && python3 scripts/validate-shortcut.py dist/Iris_unsigned.shortcut"
```

## Windows / WSL Build And Sign

From PowerShell in the repo root:

```powershell
wsl -- bash -lc "mkdir -p /tmp/cherri-iris && cd /tmp/cherri-iris && curl -L -o cherri_linux-x86_64.zip https://github.com/electrikmilk/cherri/releases/download/v2.3.0/cherri_linux-x86_64.zip && python3 - <<'PY'
import zipfile
with zipfile.ZipFile('cherri_linux-x86_64.zip') as z:
    z.extractall('.')
PY
chmod +x cherri && ./cherri --version"
```

Compile and sign with HubSign:

```powershell
wsl -- bash -lc "cd '/mnt/c/Users/affan/Fun Projects/siri' && /tmp/cherri-iris/cherri shortcuts/iris.cherri --hubsign --share=anyone --output='dist/Iris.shortcut'"
```

Mirror the signed artifact to the conventional names:

```powershell
Copy-Item -LiteralPath "dist\Iris.shortcut" -Destination "dist\Iris.signed.shortcut" -Force
Copy-Item -LiteralPath "dist\Iris.shortcut" -Destination "dist\iris.shortcut" -Force
Set-Content -LiteralPath "dist\signing-result.txt" -Value "Signing succeeded via Cherri HubSign. dist/Iris.shortcut, dist/Iris.signed.shortcut, and dist/iris.shortcut are current AEA1 signed/package-formatted artifacts." -Encoding UTF8
```

Verify the first four bytes:

```powershell
$files = @("dist\Iris.shortcut", "dist\Iris.signed.shortcut", "dist\iris.shortcut", "shortcuts\Iris_unsigned.shortcut")
$enc = [System.Text.Encoding]::GetEncoding("iso-8859-1")
foreach ($f in $files) {
  if (Test-Path -LiteralPath $f) {
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $f))
    $header = $enc.GetString($bytes, 0, [Math]::Min(4, $bytes.Length))
    $item = Get-Item -LiteralPath $f
    Write-Output "$f size=$($item.Length) header=$header modified=$($item.LastWriteTime.ToString('s'))"
  }
}
```

Expected signed output:

```text
dist\Iris.shortcut header=AEA1
dist\Iris.signed.shortcut header=AEA1
dist\iris.shortcut header=AEA1
```

Cherri also writes a local unsigned XML shortcut such as
`shortcuts/Iris_unsigned.shortcut`. That file starts with `<?xm` and is
not the normal iPhone import artifact.

## HubSign Notes

If Cherri fails with:

```text
Error: Unsupported response type: text/plain; charset=utf-8
```

retry the explicit command:

```bash
/tmp/cherri-iris/cherri shortcuts/iris.cherri --hubsign --share=anyone --output='dist/Iris.shortcut'
```

In the latest local run, a first signing attempt failed with that response-type
error, but the explicit `--hubsign --share=anyone` retry succeeded.

To compile without signing for debugging:

```bash
/tmp/cherri-iris/cherri shortcuts/iris.cherri --debug --skip-sign --output='dist/Iris.shortcut'
```

That produces unsigned XML/debug artifacts and is useful for compiler debugging,
but it is not enough for normal iPhone import.

## Local macOS Signing

On macOS, use Apple's built-in command:

```bash
shortcuts sign --mode anyone \
  --input "dist/Iris.shortcut" \
  --output "dist/Iris.signed.shortcut"
```

`--mode anyone` asks Apple/iCloud to notarize the Shortcut so anyone can import
it. This may require macOS Shortcuts/iCloud state that is not present on every CI
runner.

## GitHub Actions Signing

The `build-shortcut.yml` workflow runs on `macos-15`, installs Cherri, compiles
`shortcuts/iris.cherri`, attempts `shortcuts sign`, and uploads whichever
artifact is available.

The workflow inspects the first four bytes of the generated file:

- `AEA1`: Cherri already produced a signed/package-formatted Shortcut, so the
  workflow skips double-signing.
- anything else: the workflow attempts `shortcuts sign --mode anyone`.

If signing fails in GitHub Actions, the workflow still uploads the compiled
shortcut and the logs should capture the Apple signing error. The fallback is to
sign on a real Mac, a temporary cloud Mac, or another controlled macOS runner.

Uploaded artifacts:

- `dist/Iris.shortcut`: compiled shortcut.
- `dist/Iris.signed.shortcut`: signed shortcut, if signing succeeds.
- `dist/signing-result.txt`: one-line success/failure summary.
- `dist/signing-output.log`: raw signing command output, if produced.

## Known Open Question

The first technical spike is whether `shortcuts sign --mode anyone` works on a
fresh GitHub-hosted macOS runner without an interactive iCloud session. This file
should be updated with the result after the first public CI run.

## Fallback Order

1. GitHub-hosted macOS runner in a public repository.
2. Depot macOS trial runner.
3. Temporary cloud Mac or real Mac.
4. HubSign or a custom signing server only after accepting the trust boundary.

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .

# Load .env (if present) and export keys for this session.
$envPath = Join-Path (Get-Location) ".env"
if (Test-Path $envPath) {
	Get-Content $envPath | ForEach-Object {
		$line = $_.Trim()
		if (-not $line -or $line.StartsWith('#')) { return }
		$parts = $line -split '=', 2
		if ($parts.Count -ne 2) { return }
		$name = $parts[0].Trim()
		$value = $parts[1].Trim().Trim('"').Trim("'")
		if ($name -and $value) {
			Set-Item -Path "Env:$name" -Value $value
		}
	}
}

# Some libraries expect GOOGLE_API_KEY.
if (-not $env:GOOGLE_API_KEY -and $env:GEMINI_API_KEY) {
	$env:GOOGLE_API_KEY = $env:GEMINI_API_KEY
}

streamlit run src/omago_ai/ui/app.py

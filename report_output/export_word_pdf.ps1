param(
  [Parameter(Mandatory=$true)][string]$InputDocx,
  [Parameter(Mandatory=$true)][string]$OutputPdf,
  [Parameter(Mandatory=$true)][string]$ManifestPath
)
$ErrorActionPreference = 'Stop'
$docxPath = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
$manifestFullPath = [System.IO.Path]::GetFullPath($ManifestPath)
$word = $null
$document = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $document = $word.Documents.Open($docxPath, $false, $true)
  $pages = $document.ComputeStatistics(2) # wdStatisticPages
  $document.ExportAsFixedFormat($pdfPath, 17) # wdExportFormatPDF
  $document.Close(0)
  $document = $null
  $word.Quit()
  $word = $null
  $result = [ordered]@{
    word_rendered_pages = $pages
    pdf = $pdfPath
    pdf_bytes = (Get-Item -LiteralPath $pdfPath).Length
    renderer = 'Microsoft Word 16.0 ExportAsFixedFormat'
  }
  $result | ConvertTo-Json | Set-Content -LiteralPath $manifestFullPath -Encoding UTF8
  $result | ConvertTo-Json
}
finally {
  if ($document -ne $null) { $document.Close(0) }
  if ($word -ne $null) { $word.Quit() }
}

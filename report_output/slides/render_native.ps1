param(
  [string]$Spec = (Join-Path $PSScriptRoot 'slides_spec.json'),
  [string]$Pptx = (Join-Path $PSScriptRoot 'LSTM_GRU_課堂報告.pptx'),
  [string]$Pdf = (Join-Path $PSScriptRoot 'LSTM_GRU_課堂報告.pdf'),
  [string]$Preview = (Join-Path $PSScriptRoot 'preview')
)
$ErrorActionPreference = 'Stop'
$specData = Get-Content -LiteralPath $Spec -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Path $Preview -Force | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$initialCount = $app.Presentations.Count
$pres = $null
function Convert-HexToOleColor([string]$hex) {
  $h = $hex.TrimStart('#')
  $r = [Convert]::ToInt32($h.Substring(0,2),16)
  $g = [Convert]::ToInt32($h.Substring(2,2),16)
  $b = [Convert]::ToInt32($h.Substring(4,2),16)
  return [int]($r + 256*$g + 65536*$b)
}
try {
  $app.DisplayAlerts = 1
  $pres = $app.Presentations.Add($false)
  $pres.PageSetup.SlideWidth = [single]$specData.canvas_width
  $pres.PageSetup.SlideHeight = [single]$specData.canvas_height
  foreach ($page in $specData.slides) {
    $slide = $pres.Slides.Add($pres.Slides.Count + 1, 12)
    foreach ($e in $page.elements) {
      switch ($e.type) {
        'rect' {
          $shape = $slide.Shapes.AddShape(1, [single]$e.x, [single]$e.y, [single]$e.w, [single]$e.h)
          $shape.Fill.ForeColor.RGB = Convert-HexToOleColor $e.fill
          $shape.Line.ForeColor.RGB = Convert-HexToOleColor $e.line_color
          $shape.Line.Weight = [single]$e.line_width
        }
        'text' {
          $shape = $slide.Shapes.AddTextbox(1, [single]$e.x, [single]$e.y, [single]$e.w, [single]$e.h)
          $shape.TextFrame.MarginLeft = 0; $shape.TextFrame.MarginRight = 0
          $shape.TextFrame.MarginTop = 0; $shape.TextFrame.MarginBottom = 0
          $shape.TextFrame.WordWrap = -1
          $range = $shape.TextFrame.TextRange
          $range.Text = [string]$e.text
          $range.Font.Name = [string]$e.font_name
          try { $range.Font.NameFarEast = [string]$e.font_name } catch { }
          $range.Font.Size = [single]$e.font_size
          $range.Font.Bold = [int]([bool]$e.bold)
          $range.Font.Color.RGB = Convert-HexToOleColor $e.color
          if ($e.PSObject.Properties.Name -contains 'hyperlink' -and [string]$e.hyperlink) {
            $shape.ActionSettings(1).Action = 7
            $shape.ActionSettings(1).Hyperlink.Address = [string]$e.hyperlink
          }
          if ($e.align -eq 'center') { $range.ParagraphFormat.Alignment = 2 }
          elseif ($e.align -eq 'right') { $range.ParagraphFormat.Alignment = 3 }
          else { $range.ParagraphFormat.Alignment = 1 }
          $shape.Line.Visible = 0; $shape.Fill.Visible = 0
        }
        'line' {
          $shape = $slide.Shapes.AddLine([single]$e.x1, [single]$e.y1, [single]$e.x2, [single]$e.y2)
          $shape.Line.ForeColor.RGB = Convert-HexToOleColor $e.color
          $shape.Line.Weight = [single]$e.width
        }
        'image' {
          if ([IO.Path]::IsPathRooted([string]$e.path)) { $imagePath = [string]$e.path }
          else { $imagePath = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $Spec) $e.path)) }
          Add-Type -AssemblyName System.Drawing
          $img = [Drawing.Image]::FromFile($imagePath)
          try {
            $scale = [Math]::Min(([double]$e.w / $img.Width), ([double]$e.h / $img.Height))
            $iw = [single]($img.Width * $scale); $ih = [single]($img.Height * $scale)
            $ix = [single]($e.x + ($e.w - $iw)/2); $iy = [single]($e.y + ($e.h - $ih)/2)
          } finally { $img.Dispose() }
          $shape = $slide.Shapes.AddPicture($imagePath, 0, -1, $ix, $iy, $iw, $ih)
        }
        default { throw "Unknown element type: $($e.type)" }
      }
    }
    try { $slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = [string]$page.notes } catch { }
    $png = Join-Path $Preview ('slide_{0:D2}.png' -f $slide.SlideIndex)
    $slide.Export($png, 'PNG', 1600, 900)
  }
  $tmpPptx = Join-Path $PSScriptRoot 'native_output.pptx'
  $tmpPdf = Join-Path $PSScriptRoot 'native_output.pdf'
  $pres.SaveAs($tmpPptx, 24)
  $pres.SaveAs($tmpPdf, 32)
  $slideCount = $pres.Slides.Count
  $slideWidth = $pres.PageSetup.SlideWidth
  $slideHeight = $pres.PageSetup.SlideHeight
  $pres.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($pres); $pres = $null
  Move-Item -LiteralPath $tmpPptx -Destination $Pptx -Force
  Move-Item -LiteralPath $tmpPdf -Destination $Pdf -Force
  [ordered]@{ slides=$slideCount; width=$slideWidth; height=$slideHeight; initial_presentations=$initialCount; pptx=$Pptx; pdf=$Pdf; preview=$Preview } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'native_render_result.json') -Encoding UTF8
}
finally {
  if ($null -ne $pres) { $pres.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($pres) }
  if ($initialCount -eq 0) { $app.Quit() }
  [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}

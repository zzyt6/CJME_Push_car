param([Parameter(Mandatory=$true)][string]$DeckPath, [Parameter(Mandatory=$true)][string]$OutputStem, [switch]$PngOnly)
$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath $DeckPath).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputStem))
$exe = 'C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE'
$launched = $false
try { $office = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') } catch { $office = $null }
if ($null -eq $office) {
    $process = Start-Process -FilePath $exe -ArgumentList '/Embedding' -WindowStyle Hidden -PassThru
    $launched = $true
    for ($attempt=0; $attempt -lt 50; $attempt++) {
        try { $office = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application'); break }
        catch { Start-Sleep -Milliseconds 200 }
    }
}
if ($null -eq $office -or $office.Version -notlike '16.*' -or $office.Path -notlike '*Microsoft Office*') {
    throw 'The COM server is not Microsoft PowerPoint. Refusing another renderer.'
}
$deck = $null
try {
    $deck = $office.Presentations.Open($inputPath, $true, $false, $false)
    if (-not $PngOnly) { $deck.SaveAs($outputPath+'.pdf', 32) }
    $deck.Slides.Item(1).Export($outputPath+'.png', 'PNG', 1800, 1300)
    $charts = @()
    foreach ($shape in $deck.Slides.Item(1).Shapes) {
        if ($shape.HasChart -eq -1) {
            $chart = $shape.Chart
            $area = $chart.PlotArea
            $charts += [pscustomobject]@{
                Name=$shape.Name
                PlotRectangleMm=@(($shape.Left+$area.InsideLeft)*25.4/72; ($shape.Top+$area.InsideTop)*25.4/72; $area.InsideWidth*25.4/72; $area.InsideHeight*25.4/72)
                XMinimum=$chart.Axes(1,1).MinimumScale
                XMaximum=$chart.Axes(1,1).MaximumScale
                YMinimum=$chart.Axes(2,1).MinimumScale
                YMaximum=$chart.Axes(2,1).MaximumScale
                SeriesCount=$chart.SeriesCollection().Count
            }
        }
    }
    $report = [pscustomobject]@{Application=$office.Name; Version=$office.Version; ApplicationPath=$office.Path; InputPath=$inputPath; Charts=$charts}
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($outputPath+'.application.json') -Encoding UTF8
    $report | ConvertTo-Json -Depth 6 -Compress
} finally {
    if ($null -ne $deck) { try { $deck.Close() } catch {} }
    if ($launched -and $office.Presentations.Count -eq 0) { try { $office.Quit() } catch {} }
}

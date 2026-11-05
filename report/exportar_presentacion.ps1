$ErrorActionPreference = 'Stop'
$pptFile = Join-Path $PSScriptRoot 'presentacion_proyecto.pptx'
$pdfFile = Join-Path $PSScriptRoot 'presentacion_proyecto.pdf'
$previewDir = Join-Path $PSScriptRoot 'presentacion_recursos\vista_powerpoint'
$app = $null
$presentation = $null
$cerrarAplicacion = $false
try {
    $app = New-Object -ComObject PowerPoint.Application
    $cerrarAplicacion = ($app.Presentations.Count -eq 0)
    # Abrir sin ventana: exportación y revisión del archivo generado.
    $presentation = $app.Presentations.Open($pptFile, -1, 0, 0)
    $presentation.SaveAs($pdfFile, 32)
    $presentation.Export($previewDir, 'PNG', 1600, 900)
    Write-Output "Exportadas $($presentation.Slides.Count) diapositivas a PDF y PNG."
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $app) {
        if ($cerrarAplicacion) { $app.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
    }
}

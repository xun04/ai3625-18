$Env:CONDA_EXE = "/data2/ai3625/workspace/ai3625-18/miniconda3/bin/conda"
$Env:_CONDA_EXE = "/data2/ai3625/workspace/ai3625-18/miniconda3/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:CONDA_PYTHON_EXE = "/data2/ai3625/workspace/ai3625-18/miniconda3/bin/python"
$Env:_CONDA_ROOT = "/data2/ai3625/workspace/ai3625-18/miniconda3"
$CondaModuleArgs = @{ChangePs1 = $True}

Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs
rm -r .\HDIM\src\Generic
rm -r .\HDIM\src\FOS
rm -r .\HDIM\src\Solvers

cp -r ..\Generic .\HDIM\src
cp -r ..\FOS .\HDIM\src
cp -r ..\Solvers .\HDIM\src

New-Item -ItemType SymbolicLink -Path ..\inst\include -Name eigen3 -Value ..\..\eigen3
New-Item -ItemType SymbolicLink -Path ..\inst\include -Name boost_1_64_0 -Value ..\..\boost_1_64_0

$R_PATH = Get-Command R.exe | Select-Object source
$R_PATH_STR = $R_PATH.Source
$R_INST_CMD = "CMD INSTALL --no-lock --preclean --no-multiarch --with-keep.source HDIM"

echo "$R_PATH_STR $R_INST_CMD"

Start-Process -FilePath "$R_PATH_STR" -ArgumentList "$R_INST_CMD"

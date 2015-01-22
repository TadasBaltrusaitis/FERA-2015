@echo off

IF "SEMAINE"=="%2" set AU_loc="./AU_predictors/AU_all_SVM_simple_dyn.txt"
IF "BP4D"=="%2" set AU_loc="./AU_predictors/AU_all_SVM_simple_stat.txt"

"%~dp0./Release/AUPrediction.exe" -auloc %AU_loc% -fx 2000 -fy 2000 -ftxt %1 -rigid -asvid -simscale 0.7 -simsize 112 -oausclass %4 -oausreg %5 -oausregseg %6
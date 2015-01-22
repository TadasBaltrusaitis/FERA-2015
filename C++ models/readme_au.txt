To run the executable in windows please use (the output directories must already exist and are not created automatically):
script_au_pred.bat in_txt_file database_type empty(no feature points needed) out_occurance_loc out_intensity_loc out_intensity_segmented_loc

Example of this running (included):
script_au_pred.bat samples.txt BP4D placeholder out.class.txt out.reg.txt out.reg.seg.txt
script_au_pred.bat samples.txt SEMAINE placeholder out.class.txt out.reg.txt out.reg.seg.txt
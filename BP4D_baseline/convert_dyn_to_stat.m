clear
to_conv = dir('paper_res/*_static_combined_v_bp4d.mat');
to_conv_to = dir('paper_res/*_static_combined_v_bp4d.dat');

for i=1:numel(to_conv)
   
    trained_old = ['paper_res/' to_conv_to(i).name];
    trained_new = ['trained_SVM/' to_conv_to(i).name];
    
    copyfile(trained_old, trained_new);
    
    % append to the files
    f_from = fopen(['paper_res/' to_conv_to(i).name], 'r');        
    f = fopen(['trained_SVM/' to_conv_to(i).name], 'w');        
    
    load(['paper_res/' to_conv(i).name]);

    while(true)
        b = fread(f_from, 1, 'uint');
        if(~isempty(b))
            fwrite(f, b, 'uint');
        else
           break; 
        end
    end
        
    fseek(f, 0, 'bof');
    fwrite(f, 4, 'uint');
        
    fclose(f);
    fclose(f_from);
end
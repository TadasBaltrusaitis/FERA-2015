clear
to_conv = dir('trained/*_static.mat');
to_conv_to = dir('trained/*_static.dat');

for i=1:numel(to_conv)
   
    trained_old = ['trained/' to_conv_to(i).name];
    trained_new = ['trained_SVM/' to_conv_to(i).name];
    
    copyfile(trained_old, trained_new);
    
    % append to the files
    f_from = fopen(['trained/' to_conv_to(i).name], 'r');        
    f = fopen(['trained_SVM/' to_conv_to(i).name], 'w');        
    
    load(['trained/' to_conv(i).name]);
    
    if(numel(model.Label) > 0)
        pos_lbl = model.Label(1);
        neg_lbl = model.Label(2);
    else
        pos_lbl = 1;
        neg_lbl = 0;
    end

    while(true)
        b = fread(f_from, 1, 'uint');
        if(~isempty(b))
            fwrite(f, b, 'uint');
        else
           break; 
        end
    end
    
    fwrite(f, pos_lbl, 'float64');
    fwrite(f, neg_lbl, 'float64');
    
    fseek(f, 0, 'bof');
    fwrite(f, 4, 'uint');
        
    fclose(f);
    fclose(f_from);
end
path_tar = '/DATA_Inter/cj/localfield_100_400_50_8/';

path_localfield='/DATA_Temp/cj/QSM/NeXtQSM/train_localfield/';
%%
for index=39:100
    file_localfield = [path_localfield,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=401:500
    file_localfield = [path_localfield,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=801:900
    file_localfield = [path_localfield,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=1201:1300
    file_localfield = [path_localfield,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=1601:1700
    file_localfield = [path_localfield,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
path_test = '/DATA_Temp/cj/QSM/NeXtQSM/test_';

path_train='/DATA_Temp/cj/QSM/NeXtQSM/train_';
%%
for index=1800:1999 %
    file_totalfield = [path_train,'totalfield/totalfield_',num2str(index),'.nii.gz'];
    file_localfield = [path_train,'localfield/localfield_',num2str(index),'.nii.gz'];
    file_chimap = [path_train,'synthetic_brain/image_',num2str(index),'.nii.gz'];
    
    movefile(file_totalfield,[path_test,'totalfield/']);
    movefile(file_localfield,[path_test,'localfield/']);
    movefile(file_chimap,[path_test,'synthetic_brain/']);
    
    disp(index);
end

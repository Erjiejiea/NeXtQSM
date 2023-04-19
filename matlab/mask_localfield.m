path_localfield2 = '/DATA_Inter/cj/localfield_100_400_50_8/';
path_localfield = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield/';
path_mask = '/DATA_Temp/cj/QSM/NeXtQSM/mask/';

path_dir = '/DATA_Temp/cj/QSM/NeXtQSM/train_localfield_masked/';
%%
for index=0:0%100
    tic
    nii_localfield = load_untouch_nii([path_localfield2,'localfield_',num2str(index),'.nii.gz']);
    nii_mask = load_untouch_nii([path_mask,'mask_',num2str(index),'.nii.gz']);
    file_localfield = nii_localfield.img;
    file_mask = nii_mask.img;
    
    masked = file_localfield.*file_mask;
    figure;imshow(masked(:,:,128),[]);
    nii = make_nii(masked,[0 0 0],64);
    save_nii(nii,[path_dir,'localfield_',num2str(index),'.nii.gz']);
    
    disp(index);
    toc
end

%%
for index=401:500
    file_localfield = [path_localfield2,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=801:900
    file_localfield = [path_localfield2,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=1201:1300
    file_localfield = [path_localfield2,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
%%
for index=1601:1700
    file_localfield = [path_localfield2,'localfield_',num2str(index),'.nii.gz'];
%     nii = load_nii(file_localfield);
    movefile(file_localfield,path_tar);
    
    disp(index);
end
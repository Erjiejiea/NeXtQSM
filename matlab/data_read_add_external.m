%% add matlab toolbox
addpath(genpath('/data3/cj/QSM/_lib/'));

%% read
path_synthetic='/DATA_Temp/cj/QSM/NeXtQSM/train_synthetic_brain/';
path_added='/DATA_Temp/cj/QSM/NeXtQSM/synthetic_brain_add_external/';
path_ext='/DATA_Temp/cj/QSM/NeXtQSM/synthetic_brain_add_external/';

index = 1;
nii_img = load_untouch_nii([path_synthetic,'image_',num2str(index),'.nii.gz']); % view_nii(nii_img)
% nii_ext = load_untouch_nii([path_ext,'external_sources_',num2str(index),'.nii.gz']); % view_nii(nii_label)
nii_added = load_untouch_nii([path_added,'brain_added_',num2str(index),'.nii.gz']);
% v6:6 all
% v5:8 all x
% v4:10 all x
% v3:5 all
% v2:5 now & last
% v1:none

brain = nii_img.img;
% ext = nii_ext.img;
added = nii_added.img;

m = 1; % display type
rotkey_axial = 1;
rotkey_cronol = -1;
flipkey = 2;

%
img = added;
figure; ha = tight_subplot(1,3,[0 0],[0 0],[0 0]);

axes(ha(1)); imshow(rot90(squeeze(img(:,140,:)),rotkey_axial),[-0.1 0.1]); %colormap jet
axes(ha(1+m)); imshow(rot90(img(:,:,128),rotkey_cronol),[-0.1 0.1]);
axes(ha(1+m*2)); imshow(flip(squeeze(img(128,:,:)),flipkey),[-0.1 0.1]);

disp(['Successfully display: ',num2str(index),' !']);

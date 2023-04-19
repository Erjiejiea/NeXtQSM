%% add matlab toolbox
addpath(genpath('/data3/cj/QSM/_lib/'));

%% read
path_synthetic='/DATA_Temp/cj/QSM/NeXtQSM/train_synthetic_brain/';
path_added='/DATA_Temp/cj/QSM/NeXtQSM/synthetic_brain_add_external/';
path_localfield='/DATA_Temp/cj/QSM/NeXtQSM/train_localfield/';
path_totalfield='/DATA_Temp/cj/QSM/NeXtQSM/train_totalfield/';

index = 0;
nii_img = load_untouch_nii([path_synthetic,'image_',num2str(index),'.nii.gz']); % view_nii(nii_img)
nii_added = load_untouch_nii([path_added,'brain_added_',num2str(index),'.nii.gz']);
nii_localfield = load_untouch_nii([path_localfield,'localfield_',num2str(index),'.nii.gz']);
nii_totalfield = load_untouch_nii([path_totalfield,'totalfield_',num2str(index),'.nii.gz']);

brain = nii_img.img;
added = nii_added.img;
localfield = nii_localfield.img;
totalfield = nii_totalfield.img;

% no need to unwrap, but need to mask (done it in pycharm)

%
m = 4; % display type
rotkey_axial = 1;
rotkey_cronol = -1;
flipkey = 2;

img1 = brain;
img2 = added;
img3 = localfield;
img4 = mask_totalfiled; td = 0.1; % 0.35
img5 = unwrapped_totalfield; td2 = 0.1;
figure('Color','w'); ha = tight_subplot(3,4,[0.01 0.01],[0.01 0.01],[0.01 0.01]);
axes(ha(1)); imshow(rot90(squeeze(img1(:,140,:)),rotkey_axial),[-0.1 0.1]); %colormap jet
axes(ha(1+m)); imshow(rot90(img1(:,:,128),rotkey_cronol),[-0.1 0.1]);
axes(ha(1+m*2)); imshow(flip(squeeze(img1(128,:,:)),flipkey),[-0.1 0.1]);
axes(ha(2)); imshow(rot90(squeeze(img2(:,140,:)),rotkey_axial),[-0.1 0.1]); %colormap jet
axes(ha(2+m)); imshow(rot90(img2(:,:,128),rotkey_cronol),[-0.1 0.1]);
axes(ha(2+m*2)); imshow(flip(squeeze(img2(128,:,:)),flipkey),[-0.1 0.1]);
axes(ha(3)); imshow(rot90(squeeze(img3(:,140,:)),rotkey_axial),[-0.1 0.1]); %colormap jet
axes(ha(3+m)); imshow(rot90(img3(:,:,128),rotkey_cronol),[-0.1 0.1]);
axes(ha(3+m*2)); imshow(flip(squeeze(img3(128,:,:)),flipkey),[-0.1 0.1]);
axes(ha(4)); imshow(rot90(squeeze(img4(:,140,:)),rotkey_axial),[-td td]); %colormap jet
axes(ha(4+m)); imshow(rot90(img4(:,:,128),rotkey_cronol),[-td td]);
axes(ha(4+m*2)); imshow(flip(squeeze(img4(128,:,:)),flipkey),[-td td]);
% axes(ha(5)); imshow(rot90(squeeze(img5(:,140,:)),rotkey_axial),[-td2 td2]); %colormap jet
% axes(ha(5+m)); imshow(rot90(img5(:,:,128),rotkey_cronol),[-td2 td2]);
% axes(ha(5+m*2)); imshow(flip(squeeze(img5(128,:,:)),flipkey),[-td2 td2]);

disp(['Successfully display: ',num2str(index),' !']);

%%
figure('Color','w'); hb = tight_subplot(1,3,[0.01 0.01],[0.01 0.01],[0.01 0.01]);
n=1;
axes(hb(1)); imshow(rot90(squeeze(img1(:,140,:)),rotkey_axial),[-0.1 0.1]); %colormap jet
axes(hb(1+n)); imshow(rot90(squeeze(img3(:,140,:)),rotkey_axial),[-0.1 0.1]);
axes(hb(1+n*2)); imshow(rot90(squeeze(img4(:,140,:)),rotkey_axial),[-0.1 0.1]);

%% mask localfield
% Why unexpected ?? 

% index = 3;
% path = '/DATA_Inter/cj/localfield_100_400_50_8/';
% nii_localfield = load_untouch_nii([path,'localfield_',num2str(index),'.nii.gz']);
% localfield = nii_localfield.img;
% figure; subplot(131); imshow(rot90(squeeze(localfield(:,140,:)),rotkey_axial),[-0.1 0.1]);
% path = '/DATA_Temp/cj/QSM/NeXtQSM/mask/';
% nii = load_untouch_nii([path,'mask_',num2str(index),'.nii.gz']);
% mask = nii.img;
% subplot(132); imshow(rot90(squeeze(mask(:,140,:)),rotkey_axial),[-0.1 0.1]);
% path_local_filed2='/DATA_Temp/cj/QSM/NeXtQSM/train_localfield_masked/';
% nii_localfield = load_untouch_nii([path_localfield2,'localfield_',num2str(index),'.nii.gz']);
% localfield = nii_localfield.img;
% subplot(133); imshow(rot90(squeeze(localfield(:,140,:)),rotkey_axial),[-0.1 0.1]);

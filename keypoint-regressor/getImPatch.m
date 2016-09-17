function [patch, pose, tf] = getImPatch(img,x,y,wi,hei,patchSizHei,patchSizWi,poseGT)

% %pad the image
% if x<=0
%     padPixels = -x;
%     img = padarray(img,[0 padPixels],'replicate','pre');
%     poseGT(:,1) = poseGT(:,1) + padPixels;
%     x=1;  
% end
% 
% if (x+wi)>size(img,2)
%     padPixels = (x+wi)-size(img,2);
%     img = padarray(img,[0 padPixels],'replicate','post');
%     %wi=size(img,2);
% end
% 
% if y<=0
%     padPixels = -y;
%     img = padarray(img,[padPixels 0],'replicate','pre');
%     poseGT(:,2) = poseGT(:,2) + padPixels;
%     y=1;
% end
% 
% if (y+hei)>size(img,1)
%     padPixels = (y+hei)-size(img,1);
%     img = padarray(img,[padPixels 0],'replicate','post');
%     %hei=size(img,1);
%     
% end





%crop
crp=img(y:y+hei,x:x+wi,:);

%GT points
poseGTresc=poseGT(:,1:2);
poseGTresc(:,1)=poseGTresc(:,1)-x+1;
poseGTresc(:,2)=poseGTresc(:,2)-y+1;
%poseGTresc(:,1)=poseGTresc(:,1)./scale_fa_x;
%poseGTresc(:,2)=poseGTresc(:,2)./scale_fa_y;
%poseGTresc = reshape (poseGTresc',size(poseGTresc,1)*2,1);

%resize
%patchSizHei=100; %224
%patchSizWi = 65;
%crp=imresize(crp,[patchSiz,patchSiz]);
%scale_fa_x=wi/patchSiz;
%scale_fa_y=hei/patchSiz;

s_s = [size(crp,1) size(crp,2)];
s_t = [patchSizHei patchSizWi];
s = s_s.\s_t;
tf = [ s(2) 0 0; 0 s(1) 0; 0  0 1];
T = affine2d(tf);
patch = imwarp(crp,T);
patch=patch(1:patchSizHei,1:patchSizWi,:);
[X,Y] = transformPointsForward(T, poseGTresc(:,1),poseGTresc(:,2));
pose = [X Y];

%patch=crp;
%pose=poseGTresc;

% tf=[1/scale_fa_x 0 0; 0 1/scale_fa_y 0; 0 0 1];
% tform = maketform('affine',tf);
% [patch, grn, gcn] = transformImage(crp, poseGTresc(:,2), poseGTresc(:,1), tform);
% %crop once more being sure about the size of the image
% patch=patch(1:patchSiz,1:patchSiz,:);
% poseGTresc(:,2)=grn;
% poseGTresc(:,1)=gcn;
% pose=poseGTresc;


end
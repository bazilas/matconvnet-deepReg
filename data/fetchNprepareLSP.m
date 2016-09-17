function fetchNprepareLSP()

%Dowload LSP data and annotation
%(http://www.comp.leeds.ac.uk/mat4saj/lsp.html)

if ~exist('lsp_dataset', 'dir')
    url = 'http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip';
    fprintf('downloading %s\n', url) ;
    unzip(url) ;
    url = 'http://datasets.d2.mpi-inf.mpg.de/leonid13iccv/lsp_observer_centric.zip';
    fprintf('downloading %s\n', url) ;
    unzip(url) ;
end

cam=1;
Njo=14;

%full-body
cols = {'g', 'g','w', 'm','m','m','b','b','b','r' 'r', 'r','y','y'};
par = [2,3,4,4,4,5,8,9,13,13,10,11,14,14]; %parent of each node
coordMap=[13,13,13,13,13,13,9,10,3,4,8,11,2,5;...
    13,14, 9, 3, 4,10,8,11,2,5,7,12,1,6];
flipFlg='full';

%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

%Dataset
rootPth='lsp_dataset/';
datas='LSP';

load('lsp_observer_centric/jointsOC.mat');
joints=permute(joints,[2 1 3]);
jointsAll = joints(:,1:2,:);
loadImgName=0;

imgFmt=[rootPth 'images/im%.4d.jpg'];

%Dataset Frames
trBeg=1;
trEnd=1000;
vaBeg=1001;
vaEnd=2000;

%Bounding Box shifts
v1=20;v2=10;v3=40;v4=20;

%actor (only for multiple individuals
indiv = ones(1,vaEnd);

%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

if ~exist('LSP', 'dir')
    mkdir('LSP');
end

storeFMtTrain=['LSP/' '%s_imdbsT%daug%d.mat'];
storeFMtVal=['LSP/' '%s_imdbsV%daug%d.mat'];

%transformations per patch
Ntr=40; %30

patchHei=120; %100
patchWi =80; %65

imdb=[];

avg_hei=[];
avg_wi=[];
cnt=0;
jnts=[];
for fr=trBeg:trEnd
    
    img = imread(sprintf(imgFmt,fr)); %%%%%%NUMBERING%%%%%%
    %imshow(img); hold on;
    %text(20,20,['Frame: ' int2str(fr)],'Color',[1 1 1],'FontSize',15);
    
    poseGT=jointsAll(:,:,fr); %2D GT
    
    if size(poseGT,1)>0
        
        %ensure correct values
        poseGT(:,1) = max(1,poseGT(:,1));
        poseGT(:,1) = min(size(img,2),poseGT(:,1));
        poseGT(:,2) = max(1,poseGT(:,2));
        poseGT(:,2) = min(size(img,1),poseGT(:,2));
        
        %fit a bounding box
        x=min(poseGT(:,1));
        y=min(poseGT(:,2));
        wi=max(poseGT(:,1))-min(poseGT(:,1));
        hei=max(poseGT(:,2))-min(poseGT(:,2));
        
        %extend it
        x=round(x-v1);
        y=round(y-v2);
        wi=round(wi+v3);
        hei=round(hei+v4);
      
        if x<=0
            x=1;
            
        end
        
        if (x+wi)>size(img,2)
            wi=size(img,2)-x;
        end
        
        if y<=0
            y=1;
        end
        
        if (y+hei)>size(img,1)
            hei=size(img,1)-y;
        end
        
        %get image patch and GT
        cnt=cnt+1;
        [imdb(cnt).im, tempY,  imdb(cnt).tf] = getImPatch(img,x,y,wi,hei,patchHei,patchWi,poseGT(:,1:2));
        imdb(cnt).y = treeCoords(tempY,coordMap,patchHei,patchWi,1);
        imdb(cnt).y = reshape (imdb(cnt).y',size(imdb(cnt).y,1)*2,1);
        imdb(cnt).xy=[x;y;wi;hei];
        imdb(cnt).set=1;
        imdb(cnt).fr=fr;
        imdb(cnt).cl=1;
        imdb(cnt).flip=0;
        imdb(cnt).actor=indiv(fr);
        
        %FLIP LEFT - RIGHT
        cnt=cnt+1;
        [imdb(cnt).im, tempY] = getImFlippedPatch(imdb(cnt-1).im,tempY);
        imdb(cnt).y = treeCoords(tempY,coordMap,patchHei,patchWi,1);
        imdb(cnt).y = reshape (imdb(cnt).y',size(imdb(cnt).y,1)*2,1);
        imdb(cnt).xy=[x;y;wi;hei];
        imdb(cnt).set=1;
        imdb(cnt).fr=fr;
        imdb(cnt).cl=1;
        imdb(cnt).flip=1;
        imdb(cnt).actor=indiv(fr);
        
        %perform transformations
        x0=x;
        y0=y;
        wi0=wi;
        hei0=hei;
        
        %check if the head is down
        if poseGT(Njo,2)>poseGT(1,2)
            NtrUD=Ntr;
        else
            NtrUD=Ntr;
        end
        
        for tr=1:1:NtrUD
            
            %transform
            x=-1;
            %check that the points remain in the image plane
            tform=[];
            cntTr=1;
            imgTr=img;
            while x<=0 || y<=0 || x+wi>size(imgTr,2) || y+hei>size(imgTr,1)
                theta1=rand(1)*20;
                theta2=-rand(1)*20;
                if rand(1)>0.5
                    theta1=theta2;
                end
                
                tform = maketform('affine',[cosd(theta1) -sind(theta1) 0; sind(theta1) cosd(theta1) 0; 0 0 1]);
                
                [imgTr, grn, gcn] = transformImage(img, poseGT(:,2), poseGT(:,1), tform);
                poseGTtr(:,2)=grn;
                poseGTtr(:,1)=gcn;
                
                %fit a bounding box
                x=min(poseGTtr(:,1));
                y=min(poseGTtr(:,2));
                wi=max(poseGTtr(:,1))-min(poseGTtr(:,1));
                hei=max(poseGTtr(:,2))-min(poseGTtr(:,2));
                
                %extend it
                x=round(x-v1);
                y=round(y-v2);
                wi=round(wi+v3);
                hei=round(hei+v4);
                
                [x, y, wi, hei] = getBoxFromPoints(imgTr, x, y, wi, hei,8,2); %add translation
                
                cntTr=cntTr+1;
                
                if cntTr==6 %avoid multiple iterations / NOT THE OPTIMAL SOLUTION, FIX IT
                    
                    if x<=0
                        x=1;
                    end
                    
                    if (x+wi)>size(imgTr,2)
                        wi=size(imgTr,2)-x;
                    end
                    
                    if y<=0
                        y=1;
                    end
                    
                    if (y+hei)>size(imgTr,1)
                        hei=size(imgTr,1)-y;
                    end
                    
                end
            end
            
            %skip for outliers
            if cntTr~=6
                
                %get image patch
                cnt=cnt+1;
                [imdb(cnt).im, tempY, imdb(cnt).tf] = getImPatch(imgTr,x,y,wi,hei,patchHei,patchWi,poseGTtr);
                imdb(cnt).y = treeCoords(tempY,coordMap,patchHei,patchWi,1);
                imdb(cnt).y = reshape (imdb(cnt).y',size(imdb(cnt).y,1)*2,1);
                imdb(cnt).xy=[x;y;wi;hei];
                imdb(cnt).set=1;
                imdb(cnt).fr=fr;
                imdb(cnt).cl=1;
                imdb(cnt).flip=0;
                imdb(cnt).y = imdb(cnt).y + (2*rand(length(imdb(cnt).y),1)-1)*0.01; %add noise
                imdb(cnt).actor=indiv(fr);
                imdb(cnt).rot=tform.tdata.T;
                
                %FLIP
                cnt=cnt+1;
                [imdb(cnt).im, tempY] = getImFlippedPatch(imdb(cnt-1).im,tempY);
                imdb(cnt).y = treeCoords(tempY,coordMap,patchHei,patchWi,1);
                imdb(cnt).y = reshape (imdb(cnt).y',size(imdb(cnt).y,1)*2,1);
                imdb(cnt).xy=[x;y;wi;hei];
                imdb(cnt).set=1;
                imdb(cnt).fr=fr;
                imdb(cnt).cl=1;
                imdb(cnt).flip=1;
                imdb(cnt).y = imdb(cnt).y + (2*rand(length(imdb(cnt).y),1)-1)*0.01; %add noise
                imdb(cnt).actor=indiv(fr);
                
            end
            clear x y wi hei tform poseGTresc poseGTtr imgTr pose;
        end
        
        %Plot original image
        %rectangle('Position',[x,y,wi,hei],'EdgeColor','r');
        %for j=1:1:size(poseGT,1)
        %text(poseGT(j,1),poseGT(j,2),int2str(j),'Color','m','FontSize',22);
        %text(poseGT(8,1),poseGT(8,2)-20,int2str(actor),'Color',cols{actor},'FontSize',20);
        %line([poseGT(j,1),poseGT(par(j),1)],[poseGT(j,2),poseGT(par(j),2)],'Color',cols{actor},'LineWidth',5,'LineStyle', '--');
        %end
        
    end
    %hold off; pause();
    disp(fr);
end
save(sprintf(storeFMtTrain,datas,cam,Ntr),'imdb','-v7.3');


% for i=1:length(imdb)
%     imshow(imdb(i).im); hold on;
%     poseGT= vec2mat(imdb(i).y,2);
%     poseGT = treeCoords(poseGT,coordMap,patchHei,patchWi,0);
% 
%     for j=1:1:size(poseGT,1)
%         text(poseGT(j,1),poseGT(j,2),int2str(j),'Color','m','FontSize',22);
%         line([poseGT(j,1),poseGT(par(j),1)],[poseGT(j,2),poseGT(par(j),2)],'Color',cols{j},'LineWidth',5,'LineStyle', '-');
%     end
%     hold off; pause();
%     disp(i);
% end

clear imdb;

disp('training data done');
disp(cnt);

cnt=0;
for fr=vaBeg:vaEnd
    img = imread(sprintf(imgFmt,fr)); %%%%%%NUMBERING%%%%%%
    
    %imshow(img); hold on;
    %text(20,20,['Frame: ' int2str(fr)],'Color',[1 1 1],'FontSize',15);
    
    poseGT=jointsAll(:,:,fr);
    
    if size(poseGT,1)>0
        %fit a bounding box (whole image)
        x=1;
        y=1;
        wi = size(img,2)-1;
        hei= size(img,1)-1;
        
        %fit a bounding box
        x=min(poseGT(:,1));
        y=min(poseGT(:,2));
        wi=max(poseGT(:,1))-min(poseGT(:,1));
        hei=max(poseGT(:,2))-min(poseGT(:,2));
        
        %extend it
        x=round(x-v1);
        y=round(y-v2);
        wi=round(wi+v3);
        hei=round(hei+v4);
        
        if x<=0
            x=1;
            
        end
        
        if (x+wi)>size(img,2)
            wi=size(img,2)-x;
        end
        
        if y<=0
            y=1;
        end
        
        if (y+hei)>size(img,1)
            hei=size(img,1)-y;
        end
        
        %get image patch and GT
        cnt=cnt+1;
        [imdb(cnt).im, tempY, imdb(cnt).tf] = getImPatch(img,x,y,wi,hei,patchHei,patchWi,poseGT(:,1:2));
        imdb(cnt).y = treeCoords(tempY,coordMap,patchHei,patchWi,1);
        imdb(cnt).y = reshape (imdb(cnt).y',size(imdb(cnt).y,1)*2,1);
        imdb(cnt).xy=[x;y;wi;hei];
        imdb(cnt).set=2;
        imdb(cnt).fr=fr;
        imdb(cnt).cl=1;
        imdb(cnt).flip=0;
        imdb(cnt).actor=indiv(fr);
    end
    disp(fr);
end
save(sprintf(storeFMtVal,datas,cam,Ntr),'imdb','-v7.3');

% for i=1:length(imdb)
%     imshow(imdb(i).im); hold on;
%     poseGT= vec2mat(imdb(i).y,2);
%     poseGT = treeCoords(poseGT,coordMap,patchHei,patchWi,0);
%
%     for j=1:1:size(poseGT,1)
%         text(poseGT(j,1),poseGT(j,2),int2str(j),'Color','m','FontSize',22);
%         line([poseGT(j,1),poseGT(par(j),1)],[poseGT(j,2),poseGT(par(j),2)],'Color',cols{j},'LineWidth',5,'LineStyle', '-');
%     end
%     hold off; pause();
%     disp(i);
% end

end


function [patch, pose, tf] = getImPatch(img,x,y,wi,hei,patchSizHei,patchSizWi,poseGT)

%crop
crp=img(y:y+hei,x:x+wi,:);

%GT points
poseGTresc=poseGT(:,1:2);
poseGTresc(:,1)=poseGTresc(:,1)-x+1;
poseGTresc(:,2)=poseGTresc(:,2)-y+1;

s_s = [size(crp,1) size(crp,2)];
s_t = [patchSizHei patchSizWi];
s = s_s.\s_t;
tf = [ s(2) 0 0; 0 s(1) 0; 0  0 1];
T = affine2d(tf);
patch = imwarp(crp,T);
patch=patch(1:patchSizHei,1:patchSizWi,:);
[X,Y] = transformPointsForward(T, poseGTresc(:,1),poseGTresc(:,2));
pose = [X Y];
end

function co = treeCoords(X, map,patchSizHei,patchSizWi,flg)

co=X;
if flg==1 %front
    co(:,1) = co(:,1)./patchSizWi;
    co(:,2) = co(:,2)./patchSizHei;
elseif flg==0 %back
    co(:,1) = co(:,1)*patchSizWi;
    co(:,2) = co(:,2)*patchSizHei;
elseif flg==2 %front 3D
    for i=1:length(map)
        if map(i)~=i
            co(i,:)=X(i,:)-X(map(i),:);
        end
    end
elseif flg==3 %back 3D
    for i=1:length(map)
        if map(i)~=i
            co(i,:)=X(i,:)+co(map(i),:);
        end
    end
end
end

function [patch, y] = getImFlippedPatch(crp,poseGTresc)

%flip the image and points
crp = flip(crp,2);
poseGTresc(:,1)=size(crp,2)-poseGTresc(:,1); %change origin
temp=poseGTresc;
poseGTresc=temp;
poseGTresc(1,:)=temp(6,:);
poseGTresc(6,:)=temp(1,:);
poseGTresc(2,:)=temp(5,:);
poseGTresc(5,:)=temp(2,:);
poseGTresc(3,:)=temp(4,:);
poseGTresc(4,:)=temp(3,:);
poseGTresc(7,:)=temp(12,:);
poseGTresc(12,:)=temp(7,:);
poseGTresc(8,:)=temp(11,:);
poseGTresc(11,:)=temp(8,:);
poseGTresc(9,:)=temp(10,:);
poseGTresc(10,:)=temp(9,:);
y=poseGTresc;
patch=crp;
end

function [new_im, grn, gcn] = transformImage(I, gr, gc, TM)
[new_im, xdata, ydata] = imtransform(I, TM,'XYScale',1,'FillValues', 128);
w = xdata(2)-xdata(1) +1;
h = ydata(2)-ydata(1)+1;
scalex = size(new_im,2)/w;
scaley = size(new_im,1)/h;
go = [gc(:), gr(:)];
NPt = tformfwd(TM, go);
NPtImage(:,1) = NPt(:,1) - xdata(1) + 1;
NPtImage(:,2) = NPt(:,2) - ydata(1) + 1;
NPtImage(:,1) = NPtImage(:,1)*scalex;
NPtImage(:,2) = NPtImage(:,2)*scaley;
NPtImage = round(NPtImage);
grn = NPtImage(:,2);
gcn = NPtImage(:,1);
grn = reshape(grn, size(gr,1), size(gr,2));
gcn = reshape(gcn, size(gc,1), size(gc,2));

end

function [x, y, wi, hei, img] = getBoxFromPoints(img, x, y, wi, hei, var, flg)
%shift the bounding-box
if flg==2
    val=var;
    if rand(1)>0.5
        shift=rand(1)*val;
        x=round(x+shift);
    else
        shift=rand(1)*val;
        x=round(x-shift);
    end
    
    if rand(1)>0.5
        shift=rand(1)*val;
        y=round(y+shift);
    else
        shift=rand(1)*val;
        y=round(y-shift);
    end
    
end

%be sure that it stays within the image plane
if flg==1 || flg==2
    if x<=0
        x=1;      
    end
    
    if (x+wi)>size(img,2)
        wi=size(img,2)-x-1;
    end
    
    if y<=0
        y=1;
    end
    
    if (y+hei)>size(img,1)
        hei=size(img,1)-y-1;
    end
end

if flg==3
    if x<=0
        adj=-x;
        x  = x + adj +1;
        wi = wi - adj;
    end
    
    if (x+wi)>size(img,2)
        adj =size(img,2)-(x+wi);
        wi = wi - adj;
        x  = x  + adj;
    end
    
    if y<=0
        adj=-y;
        y  = y + adj +1;
        hei = hei - adj;
    end
    
    if (y+hei)>size(img,1)
        adj =size(img,1)-(y+hei);
        hei = hei - adj;
        y  = y  + adj;
    end
    
end

end
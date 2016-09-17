function [x, y, wi, hei, img] = getBoxFromPoints(poseGT, img, x, y, wi, hei, var, flg)

% if flg==2
%     v1=x;
%     v2=y;
%     v3=wi;
%     v4=hei;
%
%     %fit a bounding box
%     x=min(poseGT(:,1));
%     y=min(poseGT(:,2));
%     wi=max(poseGT(:,1))-min(poseGT(:,1));
%     hei=max(poseGT(:,2))-min(poseGT(:,2));
%
%     %extend it
%     x=round(x-v1);
%     y=round(y-v2);
%     wi=round(wi+v3);
%     hei=round(hei+v4);
% end


%shift the bounding-box
%rng(10);
if flg==2
    val=var;%8
    if rand(1)>0.5
        shift=rand(1)*val;
        x=round(x+shift);
        %wi=round(wi-shift);
        %x = max(x,1);
        %wi = min(x+wi,size(img,2)-1)-x;
    else
        shift=rand(1)*val;
        x=round(x-shift);
        %wi=round(wi+shift);
        %x = max(x,1);
        %wi = min(x+wi,size(img,2)-1)-x;
    end
    
    if rand(1)>0.5
        shift=rand(1)*val;
        y=round(y+shift);
        %hei=round(hei-shift);
        %y = max(y,1);
        %hei = min(y+hei,size(img,1)-1)-y;
    else
        shift=rand(1)*val;
        y=round(y-shift);
        %hei=round(hei+shift);
        %y = max(y,1);
        %hei = min(y+hei,size(img,1)-1)-y;
    end
    
end


%be sure that it stays within the image plane
if flg==1 || flg==2
    if x<=0
        x=1;
        
    end
    
    if (x+wi)>size(img,2)
        wi=size(img,2)-x-1;
        %x = x- (size(img,2) - wi);
        %if x<=0
        %    x = x+ (size(img,2) - wi);
        %    wi=size(img,2)-x;
        %end
    end
    
    if y<=0
        y=1;
    end
    
    if (y+hei)>size(img,1)
        hei=size(img,1)-y-1;
        %y = y-(size(img,1)-hei);
        %if y<=0
        %    y= y+(size(img,1)-hei);
        %    hei=size(img,1)-y;
        %end
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
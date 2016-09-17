function co = treeCoords(X, map,patchSizHei,patchSizWi,flg)

co=X;

%patchSizHei=100; %224
%patchSizWi = 65;

if flg==1 %front
    %co = zeros(size(map'));
    
    %     for i=size(map,2):-1:1
    %         if map(1,i)~=map(2,i)
    %         %if i~=14
    %             co(map(2,i),:)=X(map(2,i),:)-X(map(1,i),:);
    %             %co(i,:)=X(i,:)-X(14,:);
    %         end
    %
    %     end
    co(:,1) = co(:,1)./patchSizWi;
    co(:,2) = co(:,2)./patchSizHei;
    
    
    
    
elseif flg==0 %back
    
    %     for i=1:1:size(map,2)
    %         if map(1,i)~=map(2,i)
    %         %if i~=14
    %             co(map(2,i),:)=X(map(2,i),:)+co(map(1,i),:);
    %             %co(i,:)=X(i,:)+X(14,:);
    %         end
    %     end
    co(:,1) = co(:,1)*patchSizWi;
    co(:,2) = co(:,2)*patchSizHei;
    
elseif flg==10 %front body part
    %co(:,1) = co(:,1)./patchSizWi;
    %co(:,2) = co(:,2)./patchSizHei;
   
elseif flg==20 %back body part
    %co(:,1) = co(:,1)*patchSizWi;
    %co(:,2) = co(:,2)*patchSizHei;
    
elseif flg==2 %front 3D

    for i=1:length(map)
        if map(i)~=i
            co(i,:)=X(i,:)-X(map(i),:);
        end
    end
    %     co=X+1;
    %     co=0.5.*co;
    
elseif flg==3 %back 3D
    for i=1:length(map)
        if map(i)~=i
            co(i,:)=X(i,:)+co(map(i),:);
        end
    end
    %     co=2*X;
    %     co=co-1;
end



end
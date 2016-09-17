function Y = vl_nntukeyloss(X,c,iter,scbox, dzdy, varargin)
%A MatConvNet implementation of the robust loss from:
%Robust Optimization for Deep Regression
%V. Belagiannis, C. Rupprecht, G. Carneiro, and N. Navab,
%ICCV 2015, Santiago de Chile.
%Contact: V. Belagiannis, vb@robots.ox.ac.uk
% Copyright (C) 2016 Visual Geometry Group, University of Oxford.
% All rights reserved.
%
% This file is made available under the terms of the BSD license
% (see the COPYING file).

opts.loss = 'tukeyloss' ;
opts = vl_argparse(opts,varargin) ;

switch lower(opts.loss)
    case {'tukeyloss'}
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        %residuals
        res=(Y-X);
        
        %Median absolute deviation (MAD)
        MAD = 1.4826*mad(res',1)';
        
        %inliers (percentage of inliers)
        nonZer = round(100*sum(abs(res(:))<4.685)/numel(res));
        
        if iter<50 %(as in the paper)
        %if nonZer<70 %(similar to the above) - test it again
            MAD=MAD*7; %helps the convergence at the first iterations
        end
        
        res=bsxfun(@rdivide,res,MAD);
        c=4.685;
        
        if isempty(dzdy) %forward
            
            %tukey's beiweight function
            %(http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html)
            yt = (c.^2/6) * (1 - (1-(res./c).^2).^3);
            yt(find(abs(res)>c))=(c^2)/6;
            
            Y = sqrt(sum(yt(:)));
        else
            
            %derivatives
            tu= -1.*res.*((1-(res./c).^2).^2);
            
            Y_(1,1,:,:)= tu.*bsxfun(@lt,abs(res),c); % abs(x) < c
            
            Y = single (Y_ * dzdy);   
        end
        
    %l2loss
    case {'l2loss'}
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        res=(Y-X);
        
        n=1;
        if isempty(dzdy) %forward
            Y = (sum(res(:).^2))/numel(res);
        else
            Y_(1,1,:,:)= -1.*(Y-X);
            Y = single (Y_ * (dzdy / n) );
        end
            
    %error layer
    case {'mpe'} %mean pixel error
        X_orig = X;
        X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
        Y = [c{1,1:size(c,2)}];
        
        if isempty(dzdy) %forward
            
            %residuals
            err=abs(Y-X);
            
            %scale back to pixels
            funScale = @(A,B) A.*(B);
            err = bsxfun(funScale,err,scbox);
            Y=[];
            Y = sum(err)./size(X,1);%error per samples
            Y = sum(Y);%summed batch error
            
        else %nothing to backprop
            Y = zerosLike(X_orig) ;
        end
        
   otherwise
        error('Unknown parameter ''%s''.', opts.loss) ;
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end

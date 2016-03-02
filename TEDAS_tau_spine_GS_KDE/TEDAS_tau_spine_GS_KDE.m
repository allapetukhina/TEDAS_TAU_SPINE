%TEDAS Advanced, TEDAS Expert, TEDAS Naive and RR strategies
load('logreturnsdax6040w21122012_27112014.mat')
Date         = logreturnsdax6040w21122012_27112014(1:end,1)+693960;%to transform data from excel format to Matlab format
Date         = [Date;datenum('28-Nov-2014')];
XRET         = logreturnsdax6040w21122012_27112014(1:end,2:end-2); %creating covariate and response matrices
YRET         = logreturnsdax6040w21122012_27112014(1:end,end-1);
% IND        = YRET./100;
% FUNDS      = XRET./100;

IND          = YRET;
FUNDS        = XRET;
PORT60_40     = logreturnsdax6040w21122012_27112014(1:end,end)


%tau          = [0.05, 0.15, 0.25, 0.35, 0.50]; %set of quantile indices
lambda       = linspace(0.01,3,50); %grid of lambdas
wwidth       = 60; %moving window width
wshift       = 0;

pp           = 1;
qq           = 1;
rr           = 1;
mm           = 1;
alpha        = 0.2; %starting DCC parameters
betta        = 0.7;
garchtype    = 'GARCH';

cl = 0.01; 
z=norminv(cl,0,1); %standard normal quantile for Cornish-Fisher expansion

num_digw     = 4;
num_dig      = 4;

TargRet      = 0.8; %target return quantile level
method    = 'pow3';
% pre-allocating for the arrays to be created
clear ncap ncapp ncappp
ncap{wwidth}  = 1; 
ncapp(wwidth) = 1;
ncappp        = [];
BETA         = [];
numlam       = [];
ll           = [];
lll          = [];
VaRQ         = [];
WTS          = {};
COVM         = {};
INDEX        = {};
EstPar       = [];
Specs        = garchset('Distribution', 'Gaussian', 'Display', 'off', 'VarianceModel', 'GARCH', 'P', 1, 'Q', 1,...
    'R', 1, 'M', 1);
for l = wwidth:size(FUNDS,1)-1
    
    X     = XRET(1+wshift:l,:);
    Y     = YRET(1+wshift:l,:);
    
    if IND(l) < 0
        tau_i   = ksdensity(Y,IND(l),'function','cdf');
        tau_i   = round(tau_i.*(10^2))./(10^2);
        
        [n,k]   = size(X);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
        coeff   = L1QR(Y, X, tau_i); 
        BetaAll = round(coeff.*(10^num_digw))./(10^num_digw);
        clear XWW
        XWW     = zeros(n,k);
        for j = 1:n
            XWi      = X(j,:).*max(abs(BetaAll'),1/sqrt(n));
            XWW(j,:) = XWi;
        end

        BetaLam = round(BetaAll.*(10^num_digw))./(10^num_digw);
        %rule-of-thumb lambda (see the text) 
        Lambda  = 0.25*sqrt(sum(abs(BetaLam) > 0))*log(k)*(log(n))^(0.1/2);
        Coeff   = L1QR(Y, XWW, tau_i, Lambda);

        %calculating the final Adaptive Lasso coefficients
        COEFF   = Coeff.*max(abs(BetaLam),1/sqrt(n));
        BETAFIN = round(COEFF.*(10^num_digw))./(10^num_digw);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ind     = find(BETAFIN(:,1) < 0);
        RetMat  = FUNDS(1:l,ind);
        [nn,kk] = size(RetMat);
        w0      = ones(1,size(RetMat,2))./size(RetMat,2);
        MeanRet = mean(RetMat)';
        %Cmat   = eye(nn) - (1/nn)*ones(nn,1)*ones(1,nn);
        P       = RetMat;

        if size(P,2) > 1
            count     = 0;
            err_count = 0;
            while count == err_count
                try 
                [~,F,A,~,estp,SigMat] = icatvd_init(P,method);
                catch err    
                    if strcmpi(method,'pow3') == 1
                        method = 'tanh';
                    elseif strcmpi(method,'tanh') == 1 
                        method = 'gauss';
                    elseif strcmpi(method,'gauss') == 1
                        method = 'skew';
                    elseif strcmpi(method,'skew') == 1
                        method = 'pow3';
                    end
                    err_count = err_count + 1;
                end
                count = count + 1;
            end
            [~,M2,M3,M4,skkurt,EstP]    = icatvd(P,estp,SigMat,F,A,'forecast'); 
            [Ht,numfactors]             = ogarch(P,1,1,1,1,varthresh);
        elseif size(P,2) == 1                   
            [Parameters, ~, ~, ~, ~, ~] = garchfit(Specs,P);
            [HForec,~]                  = garchpred(Parameters,P,1);
            M2                          = HForec^2;
            M3                          = skewness(P);
            M4                          = kurtosis(P);
        else
            cap{l}   = sum(cell2mat(cap(l)));
            VaR      = VaRQ(end);
            cap{l+1} = cap{l};
            capp     = cap{l+1};
            cappp    = [cappp,capp] 
            VaRQ     = [VaRQ,VaR]; 
            wshift   = wshift + 1; 
            ll       = [ll,l];
            continue                     
        end
        CovMat    = M2;
        wub       = ones(length(w0),1);
        wlb       = zeros(length(w0),1);
        Aeq       = ones(1,length(w0));
        beq       = 1;
        AA        = -MeanRet';
        bb        = -quantile(MeanRet,TargRet);
        cap{l}    = sum(cell2mat(cap(l)));
        ucap{l}   = sum(cell2mat(ucap(l)));

       
      
      
        
        %check the CVaR output 
        if abs(VaR*cap{l}) > cap{l}
            [n,k]    = size(X);
            CoefTemp = zeros(k,length(lambda));
            SIC      = zeros(1,length(lambda));
            for i = 1:length(lambda)
                beta          = L1QR(Y, X, tau_i, lambda(i));
                beta          = round(beta.*(10^num_digw))./(10^num_digw);
                SIC(i)        = n*log(n^(-1)*(sum(qrRho(Y - X*beta,tau_i)))) + log(n)*(length(beta(beta ~= 0)));
                CoefTemp(:,i) = beta;
            end
            bicmin  = find(SIC == min(SIC));
            if length(bicmin) > 1
                bicmin = bicmin(end);
            end
            BetaAll = CoefTemp(:,bicmin); 
            clear XWW
            XWW = zeros(n,k);
            for j = 1:n
                XWi = X(j,:).*max(abs(BetaAll'),1/sqrt(n));
                XWW(j,:)  = XWi;
            end
            BetaLam = round(BetaAll.*(10^num_digw))./(10^num_digw);
            %rule-of-thumb lambda (see the text) 
            Lambda  = 0.25*sqrt(sum(abs(BetaLam) > 0))*log(k)*(log(n))^(0.1/2);
            Coeff   = L1QR(Y, XWW, tau_i, Lambda);
            %calculating the final Adaptive Lasso coefficients
            COEFF   = Coeff.*max(abs(BetaLam),1/sqrt(n));
            BETAFIN = round(COEFF.*(10^num_digw))./(10^num_digw);

            ind     = find(BETAFIN(:,1) < 0);
            if length(ind) > 10
                ind = ind(1:10);
            end                   
            RetMat  = FUNDS(1:l,ind);
            [nn,kk] = size(RetMat);
            w0      = ones(1,size(RetMat,2))./size(RetMat,2);
            MeanRet = mean(RetMat)';
            %Cmat    = eye(nn) - (1/nn)*ones(nn,1)*ones(1,nn);
            P       = RetMat;
            if size(P,2) > 1
            count     = 0;
            err_count = 0;
            while count == err_count
                try 
                [~,F,A,~,estp,SigMat] = icatvd_init(P,method);
                catch err    
                    if strcmpi(method,'pow3') == 1
                        method = 'tanh';
                    elseif strcmpi(method,'tanh') == 1 
                        method = 'gauss';
                    elseif strcmpi(method,'gauss') == 1
                        method = 'skew';
                    elseif strcmpi(method,'skew') == 1
                        method = 'pow3';
                    end
                    err_count = err_count + 1;
                end
                count = count + 1;
            end
                [~,M2,M3,M4,skkurt,EstP]    = icatvd(P,estp,SigMat,F,A,'forecast');
                [Ht,numfactors]             = ogarch(P,1,1,1,1,varthresh);
            elseif size(P,2) == 1                   
                [Parameters, ~, ~, ~, ~, ~] = garchfit(Specs,P);
                [HForec,~]                  = garchpred(Parameters,P,1);
                M2                          = HForec^2;
                M3                          = skewness(P);
                M4                          = kurtosis(P);
            else
                cap{l}   = sum(cell2mat(cap(l)));
                VaR      = VaRQ(end);
                cap{l+1} = cap{l};
                capp     = cap{l+1};
                cappp    = [cappp,capp] 
                VaRQ     = [VaRQ,VaR]; 
                wshift   = wshift + 1; 
                ll       = [ll,l];
            continue                     
            end
            CovMat     = M2;
            wub        = ones(length(w0),1);
            wlb        = zeros(length(w0),1);
            Aeq        = ones(1,length(w0));
            beq        = 1;
            AA         = -MeanRet';
            bb         = -quantile(MeanRet,TargRet);
            cap{l}     = sum(cell2mat(cap(l)));

            [wts,VaR]  = fmincon(@(w) -w*MeanRet + (1 + (z*((1/((w*M2*w')^(3/2)))*((w*M3*kron(w',w'))/6))) + ((( (1/((w*M2*w')^2))*...
                (w*M4*kron(kron(w',w'),w')) - 3)/24))*(z^2 - 1) - ((((1/((w*M2*w')^(3/2)))*...
                (((w*M3*kron(w',w')))))^2)/36)*(2*z^2 - 1) )*(sqrt(w*M2*w'))*factor_a,...
                w0,AA,bb,Aeq,beq,wlb,wub,[],options);
            wts        = round(wts.*(10^num_digw))./(10^num_digw);
            
            [wgt,PCov] = fmincon(@(w)((sqrt(w*Ht*w'))),w0,AA,bb,Aeq,beq,wlb,wub,[],options);
            wgt        = round(wgt.*(10^num_digw))./(10^num_digw);

            [uwts,EU]  = fmincon(@(w) exp(-eta*(w*MeanRet))*(1 + (eta^2/2)*(w*M2*w') - (eta^3/factorial(3))*...
                        (w*M3*kron(w',w')) + (eta^4/factorial(4))*(w*M4*kron(kron(w',w'),w')) ),...
                        w0,AA,bb,Aeq,beq,wlb,wub,[],options);
            uwts       = round(uwts.*(10^num_digw))./(10^num_digw);

        end

        
        %equal weights: TEDAS Naive
        ncap{l}    = (repmat(sum(cell2mat(ncap(l)))/(length(ind)),1,length(ind)));
        ncap{l+1}  = sum( cell2mat(ncap(l)).*(1 + FUNDS(l+1,ind)) );
        ncapp      = ncap{l+1};    
        ncappp     = [ncappp,ncapp];
                 
    else

        tau_i   = 0.7;        
        [n,k]   = size(X);      
        coeff   = L1QR(Y, X, tau_i); 
        BetaAll = round(coeff.*(10^num_digw))./(10^num_digw);
        clear XWW
        XWW     = zeros(n,k);
        for j = 1:n
            XWi = X(j,:).*max(abs(BetaAll'),1/sqrt(n));
            XWW(j,:)  = XWi;
        end

        BetaLam = round(BetaAll.*(10^num_digw))./(10^num_digw);
        %rule-of-thumb lambda (see the text) 
        Lambda  = 0.25*sqrt(sum(abs(BetaLam) > 0))*log(k)*(log(n))^(0.1/2)
        Coeff   = L1QR(Y, XWW, tau_i, Lambda);

        %calculating the final Adaptive Lasso coefficients
        COEFF   = Coeff.*max(abs(BetaLam),1/sqrt(n));
        BETAFIN = round(COEFF.*(10^num_digw))./(10^num_digw);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ind     = find(BETAFIN(:,1) > 0);
        RetMat  = FUNDS(1:l,ind);
        [nn,kk] = size(RetMat);
        w0      = ones(1,size(RetMat,2))./size(RetMat,2);
        MeanRet = mean(RetMat)';
        %Cmat    = eye(nn) - (1/nn)*ones(nn,1)*ones(1,nn);
        P       = RetMat;

        if size(P,2) > 1
            count     = 0;
            err_count = 0;
            while count == err_count
                try 
                [~,F,A,~,estp,SigMat] = icatvd_init(P,method);
                catch err    
                    if strcmpi(method,'pow3') == 1
                        method = 'tanh';
                    elseif strcmpi(method,'tanh') == 1 
                        method = 'gauss';
                    elseif strcmpi(method,'gauss') == 1
                        method = 'skew';
                    elseif strcmpi(method,'skew') == 1
                        method = 'pow3';
                    end
                    err_count = err_count + 1;
                end
                count = count + 1;
            end         
            [~,M2,M3,M4,skkurt,EstP]    = icatvd(P,estp,SigMat,F,A,'forecast');
            [Ht,numfactors]             = ogarch(P,1,1,1,1,varthresh);
        elseif size(P,2) == 1                   
            [Parameters, ~, ~, ~, ~, ~] = garchfit(Specs,P);
            [HForec,~]                  = garchpred(Parameters,P,1);
            M2                          = HForec^2;
            M3                          = skewness(P);
            M4                          = kurtosis(P);
        else
            cap{l}   = sum(cell2mat(cap(l)));
            VaR      = VaRQ(end);
            cap{l+1} = cap{l};
            capp     = cap{l+1};
            cappp    = [cappp,capp] 
            VaRQ     = [VaRQ,VaR] 
            wshift   = wshift + 1 
            ll       = [ll,l]
            continue                     
        end
        CovMat = M2;
        wub    = ones(length(w0),1);
        wlb    = zeros(length(w0),1);
        Aeq    = ones(1,length(w0));
        beq    = 1;
        AA     = -MeanRet';
        bb     = -quantile(MeanRet,TargRet);
        cap{l} = sum(cell2mat(cap(l)));

            
        if abs(VaR*cap{l}) > cap{l}
            [n,k]    = size(X);
            CoefTemp = zeros(k,length(lambda));
            SIC      = zeros(1,length(lambda));
            for i = 1:length(lambda)
                beta          = L1QR(Y, X, tau_i, lambda(i));
                beta          = round(beta.*(10^num_digw))./(10^num_digw);
                SIC(i)        = n*log(n^(-1)*(sum(qrRho(Y - X*beta,tau_i)))) + log(n)*(length(beta(beta ~= 0)));
                CoefTemp(:,i) = beta;
            end
            bicmin  = find(SIC == min(SIC));
            if length(bicmin) > 1
                bicmin = bicmin(end);
            end
            BetaAll = CoefTemp(:,bicmin); 
            clear XWW
            XWW = zeros(n,k);
            for j = 1:n
                XWi = X(j,:).*max(abs(BetaAll'),1/sqrt(n));
                XWW(j,:)  = XWi;
            end
            BetaLam = round(BetaAll.*(10^num_digw))./(10^num_digw);
            %rule-of-thumb lambda (see the text) 
            Lambda  = 0.25*sqrt(sum(abs(BetaLam) > 0))*log(k)*(log(n))^(0.1/2);
            Coeff   = L1QR(Y, XWW, tau_i, Lambda);
            %calculating the final Adaptive Lasso coefficients
            COEFF   = Coeff.*max(abs(BetaLam),1/sqrt(n));
            BETAFIN = round(COEFF.*(10^num_digw))./(10^num_digw);

            ind     = find(BETAFIN(:,1) > 0);
            RetMat  = FUNDS(1:l,ind);
            [nn,kk] = size(RetMat);
            w0      = ones(1,size(RetMat,2))./size(RetMat,2);
            MeanRet = mean(RetMat)';
            %Cmat    = eye(nn) - (1/nn)*ones(nn,1)*ones(1,nn);
            P       = RetMat;

%             %estimating the icatvd model to estimate the time-changing
%             %distribution structure
%             if size(P,2) > 1
%             count     = 0;
%             err_count = 0;
%             while count == err_count
%                 try 
%                 [~,F,A,~,estp,SigMat] = icatvd_init(P,method);
%                 catch err    
%                     if strcmpi(method,'pow3') == 1
%                         method = 'tanh';
%                     elseif strcmpi(method,'tanh') == 1 
%                         method = 'gauss';
%                     elseif strcmpi(method,'gauss') == 1                        method = 'skew';
% 
%                     elseif strcmpi(method,'skew') == 1
%                         method = 'pow3';
%                     end
%                     err_count = err_count + 1;
%                 end
%                 count = count + 1;
%             end
%                 [~,M2,M3,M4,skkurt,EstP]    = icatvd(P,estp,SigMat,F,A,'forecast');  
%                 [Ht,numfactors]             = ogarch(P,1,1,1,1,varthresh);
%             elseif size(P,2) == 1                   
%                 [Parameters, ~, ~, ~, ~, ~] = garchfit(Specs,P);
%                 [HForec,~]                  = garchpred(Parameters,P,1);
%                 M2                          = HForec^2;
%                 M3                          = skewness(P);
%                 M4                          = kurtosis(P);
%             else
%                 cap{l}   = sum(cell2mat(cap(l)));
%                 VaR      = VaRQ(end);
%                 cap{l+1} = cap{l};
%                 capp     = cap{l+1};
%                 cappp    = [cappp,capp] 
%                 VaRQ     = [VaRQ,VaR]; 
%                 wshift   = wshift + 1; 
%                 ll       = [ll,l];
%                 continue                     
%             end
%             CovMat = M2;
%             wub    = ones(length(w0),1);
%             wlb    = zeros(length(w0),1);
%             Aeq    = ones(1,length(w0));
%             beq    = 1;
%             AA     = -MeanRet';
%             bb     = -quantile(MeanRet,TargRet);
%             cap{l} = sum(cell2mat(cap(l)));
% 
%             [wts,VaR]  = fmincon(@(w) -w*MeanRet + (1 + (z*((1/((w*M2*w')^(3/2)))*((w*M3*kron(w',w'))/6))) + ((( (1/((w*M2*w')^2))*...
%                 (w*M4*kron(kron(w',w'),w')) - 3)/24))*(z^2 - 1) - ((((1/((w*M2*w')^(3/2)))*...
%                 (((w*M3*kron(w',w')))))^2)/36)*(2*z^2 - 1) )*(sqrt(w*M2*w'))*factor_a,...
%                 w0,AA,bb,Aeq,beq,wlb,wub,[],options);
%             wts        = round(wts.*(10^num_digw))./(10^num_digw);
%             
%             [wgt,PCov] = fmincon(@(w)((sqrt(w*Ht*w'))),w0,AA,bb,Aeq,beq,wlb,wub,[],options);
%             wgt        = round(wgt.*(10^num_digw))./(10^num_digw);
% 
%             [uwts,EU]  = fmincon(@(w) exp(-eta*(w*MeanRet))*(1 + (eta^2/2)*(w*M2*w') - (eta^3/factorial(3))*...
%                         (w*M3*kron(w',w')) + (eta^4/factorial(4))*(w*M4*kron(kron(w',w'),w')) ),...
%                         w0,AA,bb,Aeq,beq,wlb,wub,[],options);
%             uwts       = round(uwts.*(10^num_digw))./(10^num_digw);
% 
 end
                   
        %portfolio value appreciation
       
        ncap{l}    = (repmat(sum(cell2mat(ncap(l)))/(length(ind)),1,length(ind)));
        ncap{l+1}  = sum( cell2mat(ncap(l)).*(1 + FUNDS(l+1,ind)) );
        ncapp      = ncap{l+1};    
        ncappp     = [ncappp,ncapp];
    end
            
    BETA   = [BETA,BETAFIN];
    WTS    = [WTS;wts];
    UWTS   = [UWTS;uwts];
    WGT    = [WGT;wgt];%array of asset weights estimated at different steps in the loop
    COVM   = [COVM,M2];
    COVMOG = [COVMOG,Ht];%array of time-varying covariance matrices estimated at different steps in the loop
    SKM    = [SKM,M3];
    KURTM  = [KURTM,M4];
    INDEX  = [INDEX,ind];
    EXPU   = [EXPU,EU];
    ESTP   = [ESTP,EstP];
    lll    = [lll,l];
    SKKURT = [SKKURT,skkurt];
    VaRQ   = [VaRQ,VaR]; %Value-at-Risk vector
    PCovOG = [PCovOG,PCov];

    wshift = wshift + 1

    
end
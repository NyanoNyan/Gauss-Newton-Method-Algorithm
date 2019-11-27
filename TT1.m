clc, clear all
%% Load sunspot data
% Separates the year and real numbers
load sunspot.dat
TSS_Limit = 0.02;
year=sunspot(:,1); relNums=sunspot(:,2); 
%% Gets the mean and std of x, Normalizes data
ynrmv=mean(relNums(:)); sigy=std(relNums(:)); 
nrmY=relNums;  
ymin=min(nrmY(:)); ymax=max(nrmY(:)); 
relNums=2.0*((nrmY-ymin)/(ymax-ymin)-0.5);

%%  create a matrix of lagged values for a time series vector
Ss=relNums';
idim=10; % input dimension
odim=length(Ss)-idim; % output dimension

for i=1:odim
   y(i)=Ss(i+idim);
   for j=1:idim
       x(i,j) = Ss(i-j+idim); %x(i,idim-j+1) = Ss(i-j+idim);
   end
end

Patterns = x'; Desired = y; NHIDDENS = 5; prnout=Desired;
[NINPUTS,NPATS] = size(Patterns); [NOUTPUTS,NP] = size(Desired);

% load Weights1
% load Weights2
Weights1 = 0.5*(rand(NHIDDENS,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NHIDDENS)-0.5); 
W11=Weights1;
W22=Weights2;
LearnRate = 1; Momentum = 0; DerivIncr = 0; deltaW1 = 0; deltaW2 = 0;
Inputs1= [ones(1,NPATS);Patterns]'; %Inputs1 = [ones(1,NPATS); Patterns];
uu=0.001; 

NumNodes=5;Out=1;

    % Doing for each pattern

for epoch1=1:200
    % Clearing of delta Weights for each epoch
    dEdW1=zeros(5,11);
    dEdW2=zeros(1,6);
                    % Getting the Jacobian with Beta=1,HiddenBeta=1
                    if epoch1==1
                        for p=1:NPATS    
                            %% Forward Propogation
                            for i=1:NumNodes
                                activation=0;
                                for j=1:size(Inputs1,2)
                                    activation = activation + Weights1(i,j)*Inputs1(p,j);
                                    
                                    hiddenLay(i) = 1.0/(1.0 + exp(-activation)); 
                                end

                            end
                            if p==1
                            hiddenLay = [ones(1) hiddenLay];
                            end

                            % Hidden to Output
                                for i = 1: Out
                                    SigOut = 0;
                                    for w = 1:NumNodes+1
                                        SigOut =  hiddenLay(w)*Weights2(1,w) + SigOut;
                                    end
                                end
                                SumOut = SigOut;

                              %% Backward Propogation
                                  %Error in Output Nodes
                            for i = Out:-1:1
                                    err = (prnout(p) - SumOut);
                                BetaOut =  1;
                            end

                                % Error in Hidden Nodes
                            for  j = NumNodes+1:-1:1
                                        bHidden(j) = 0;
                                        for l = 1: Out 
                                            bHidden(j) = Weights2(j) * BetaOut(l);
                                            bHidden(j) = hiddenLay(j) * (1-hiddenLay(j))* bHidden(j);
                                        end
                            end
%                                             bHidden=1;
                        %% Jacobian Time
                                % Not sure about the formula also additional bHidden for bias
                                    % Input to Hidden
                                            for i =1:size(Inputs1,2)
                                                for j=1:NumNodes
                                                JbI_O(j,i)=bHidden(1,j)*Inputs1(p,i);
                                                end
                                            end
                                    % Hidden to Output
                                            for jj=1:NumNodes+1
                                                jbH_O(1,jj)= (BetaOut)*hiddenLay(jj);
                                            end
                                      Jtest = reshape(JbI_O',[],1);

                                     Jacob(p,:)=[Jtest' jbH_O(1,:)]; 
                        end
                    end
                    testing=Jacob'*Jacob;
                    
  %% Second NN
     for p=1:NPATS    
                %% Forward Propogation
                for i=1:NumNodes
                    activation=0;
                    for j=1:size(Inputs1,2)
                        activation = activation + Weights1(i,j)*Inputs1(p,j);
                        hiddenLay(i) = 1.0/(1.0 + exp(-activation)); 
                    end
                end
                
                % Hidden to Output
                    for i = 1: Out
                        SigOut = 0;
                        for w = 1:NumNodes+1
                            SigOut =  hiddenLay(w)*Weights2(1,w) + SigOut;
                        end
                    end
                    SumOut = SigOut;
                    OutStore(p)=SumOut;
                  %% Backward Propogation
                      %Error in Output Nodes
                for i = Out:-1:1
                        err = (prnout(p) - SumOut);
                    BetaOut =  err;
                    tt(p)=BetaOut;
                end

                    % Error in Hidden Nodes
                for  j = NumNodes+1:-1:1
                            bHidden(j) = 0;
                            for l = 1: Out 
                                bHidden(j) = Weights2(j) * BetaOut(l);
                                bHidden(j) = (hiddenLay(j) * (1-hiddenLay(j))* bHidden(j));
                            end
                end
               if epoch1>1
                      %% Jacobian Time
            % Not sure about the formula also additional bHidden for bias
                % Input to Hidden
                        for i =1:size(Inputs1,2)
                            for j=1:NumNodes
                            JbI_O(j,i)=(bHidden(1,j)*Inputs1(p,i));
                            end
                        end
                % Hidden to Output
                        for jj=1:NumNodes+1
                            jbH_O(1,jj)= (BetaOut)*hiddenLay(jj);
                        end
                  Jtest = reshape(JbI_O',[],1);
                  
                 Jacob(p,:)=[Jtest' jbH_O(1,:)]; 
               end
                   
         % Update for Input to Hidden
            for i = 1:NumNodes            
                for j = 1:size(Inputs1,2)
                   dEdW1(i,j)=dEdW1(i,j)+bHidden(i)*Inputs1(p,j);
                end            
            end
            
                for i = 1:Out
                    for j = 1:NumNodes   
                       dEdW2(i,j)=dEdW2(i,j)+BetaOut(i)*hiddenLay(j);
                    end  
                end      
                
     end
         %% Jacobian Calculation
                % Update for Hidden to Output
         Hess=(Jacob'*Jacob+uu*eye(61));
         invDiagHess=diag(1./(Hess))/NPATS/5;
         LR1=invDiagHess(1:55);
         LR11 = reshape(LR1,[],11);
         LR2=invDiagHess(56:61);
         
       TSS =   sum((tt).^2);
       MSE(epoch1) =  sum((tt).^2);


                %% Weight update Batch Mode
            for i = 1:NumNodes            
                for j = 1:size(Inputs1,2)
                   Weights1(i,j) = Weights1(i,j)+ LR11(i,j) * dEdW1(i,j); 
                end            
            end 
                for i = 1:Out
                    for j = 1:NumNodes+1  
                       Weights2(i,j) = Weights2(i,j) + LR2(i) * dEdW2(i,j); 
                    end  
                end    
     tt;
     uu;
     fprintf('Epoch %3d:  Error = %f\n',epoch1,TSS);
     if TSS < TSS_Limit, break, end
end
plot([11:278],Desired(11:278),[11:278],OutStore(11:278));
title('Sunspot Data')

plot(MSE);hold on;

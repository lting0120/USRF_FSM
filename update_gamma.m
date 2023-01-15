function gamma = update_gamma(L_sum,W,M)
% order = size(L,2);
m = size(L_sum,3);
% d = size(L{1},2);
% n = size(L{1},1);
f = zeros(m,1);
% L_sum = zeros(n,d,m);

% for kk = 1:order
%     L_temp = L{kk};
%     for tt = 1:m
%         L_sum(:,:,tt) = L_sum(:,:,tt)+L_temp(:,:,tt);
%     end
% end

for i = 1:m
    f(i) = trace(W'*L_sum(:,:,i)*W);
end
options = optimset('Display','off');
gamma = quadprog(M/2,-f,[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
end

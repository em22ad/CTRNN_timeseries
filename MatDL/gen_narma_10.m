function s=gen_narma_10(sz)

order=10;

u=rand(sz+order,1)*0.49999;

for i=1:order
    s(i)=0.1;
end

for i=order+1:sz
    s(i)=(0.3*s(i-1))+(0.05*s(i-1)*sum(s((i-1)-(order-1):i-1)))+1.5*u((i-1)-(order-1))*u(i-1)+0.1;
end
    
plot(1:sz,s)
data = read.table("ch1_feature.txt",sep="\t",header=T)
library(randomForest)
library(kernlab)
library(caret)
library(nnet)


data = as.matrix(data[,c(1,16:ncol(data))])
	

class(data) = "numeric"
for (k in 1:ncol(data)){
	tmp = data[,k]
	myavg = mean(tmp,na.rm=T)
	tmp[is.na(tmp)] = myavg
	data[,k] = tmp
}


####### Training #########
par(mfrow=c(2,2),mar=c(3,3,3,3))
fit <- randomForest(data[,2:ncol(data)],data[,1], ntree=1000,importance=TRUE)
pre <- predict(fit)   # prediction from OOB !!!
plot(data[,1],pre,pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="RF - train")
m = cor.test(pre,data[,1])
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")

fit <- knnreg(data[,c(2:ncol(data))],data[,1], k=3)
pre <- predict(fit,data[,c(2:ncol(data))])
plot(data[,1],pre,pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="KNN - train")
m = cor.test(pre,data[,1])
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")

fit <- ksvm(data[,c(2:ncol(data))],data[,1])
pre <- predict(fit,data[,c(2:ncol(data))])
plot(data[,1],pre,pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="SVM - train")
m = cor.test(pre,data[,1])
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")

fit <- nnet(data[,2:ncol(data)],data[,1],size=12, maxit=500, linout=T, decay=0.01)
pre <- predict(fit,data[,2:ncol(data)])
plot(data[,1],pre,pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="NNET - train")
m = cor.test(pre,data[,1])
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")


####### Manural 10-fold cross #######
kk=10
res.rf = NULL
res.knn = NULL	
res.svm = NULL
res.net = NULL
	
asiz = ceiling(nrow(data)/kk)
data = data[sample(1:nrow(data),nrow(data)),]

for(k in 1:kk)
{
	se = ((k-1)*asiz+1):min(k*asiz, nrow(data))
	tr = data[-se,]			
	te <- data[se,]	
	rownames(tr) <- NULL
	rownames(te) <- NULL
	class(tr)="numeric"
	class(te) = "numeric" 
		
	tr.label <- tr[,1]
	te.label <- te[,1]
	
	tr <- tr[,2:ncol(tr)]
	te <- te[,2:ncol(te)]
	
	fit=randomForest(tr,tr.label,importance=TRUE,ntree=1000) 
	pre = predict(fit,te)
	tmp = cbind(te.label,pre)
	res.rf = rbind(res.rf, tmp)
	
	fit=knnreg(tr,tr.label,k=4) 
	pre = predict(fit,te)
	tmp = cbind(te.label,pre)
	res.knn = rbind(res.knn, tmp)

	fit= ksvm(tr, tr.label)
	pre = predict(fit,te)
	tmp = cbind(te.label, pre)
	res.svm = rbind(res.svm, tmp)
	
	fit=nnet(tr,tr.label,size=12, maxit=500, linout=T, decay=0.01) 
	pre = predict(fit,te)
	tmp = cbind(te.label,pre)
	res.net = rbind(res.net, tmp)		
	
}

par(mfrow=c(2,2),mar=c(3,3,3,3))
 plot(res.rf[,1],res.rf[,2],pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="RF - test")
 m = cor.test(res.rf[,1],res.rf[,2])  # or report MSE ..... 
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")
 plot(res.knn[,1],res.knn[,2],pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="KNN - test")
  m = cor.test(res.knn[,1],res.knn[,2])
 legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")
 plot(res.svm[,1],res.svm[,2],pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="SVM - test")
  m = cor.test(res.svm[,1],res.svm[,2])
 legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")
 plot(res.net[,1],res.net[,2],pch=20,col="blue",xlim=c(-100,100),ylim=c(-100,100),xlab="True",ylab="Predict",main="NNET - test")
 m = cor.test(res.net[,1],res.net[,2])
legend("topright",legend=paste(m$estimate,"\n",m$p.value,sep=""),bty="n")


######## Deep learning #######   
#### h2o package contains other machine-learning techniques, besides deep learning..... 
#### overwhelming overfitting, need to tune parameters or apply L1/L2 regularization

data = read.table("ch1_feature.txt",sep="\t",header=T)
data = data.frame(data[,c(1,16:ncol(data))])
for (i in 1:ncol(data)){
	data[i,] = as.numeric(data[i,])
}

library(h2o)
h2oServer <- h2o.init(nthreads=-1)

data.hex <- as.numeric(as.h2o(object = data))
data.hex[,1] = as.numeric(data.hex[,1])

regression_model <- h2o.deeplearning(x=2:38, y=1, training_frame=data.hex, 
                                      hidden=c(50,50), epochs=100, 
                                      nfolds=5)
regression_model

h2o.performance(regression_model,data.hex)



logistic <- function(x,y,iterations=1e5, burnin=1e4, adapt=1e4, thin=100, b_start=as.matrix(rnorm(ncol(x)))){
	.Call( "logistic", x,y, iterations, burnin, adapt, thin, b_start, PACKAGE = "gibbsLogistic" )
}


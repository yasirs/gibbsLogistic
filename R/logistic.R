
logistic <- function(x,y,iterations=1e5, burnin=1e4, adapt=1e4, thin=100){
	.Call( "logistic", x,y, iterations, burnin, adapt, thin, PACKAGE = "gibbsLogistic" )
}


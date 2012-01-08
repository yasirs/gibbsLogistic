
logistic <- function(x,y){
	.Call( "logistic", x,y, PACKAGE = "gibbsLogistic" )
}


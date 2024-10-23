##############################################################################
# PROGRAM: lotto-number-generator.R
# PURPOSE: 
# WRITTEN BY: Phillip Hungerford
# DATE: 07/07/2020
##############################################################################

fun <- function(){
    #"""
    # Randomly picks 7 numbers from the bucket of 1 -> 45 for desired num of 
    # games. 
    #"""
    
    #=========================================================================
    # Asks for number of games
    x <- readline("How many games do you want to play?") 
    x <- as.numeric(unlist(strsplit(x, ",")))
    
    #=========================================================================
    # Create bucket of balls (numbers 1-45)
    bucket <- 1:45
    
    #=========================================================================
    # Randomly picks 7 numbers based on desired games
    for (i in 1:x){
        cat("Game ", i, ": ")
        # Draw numbers (replacement = F so no repeats)
        drawn <- sample(bucket, 7, replace = F)
        cat(drawn, "\n")
    }
}
##############################################################################
# run interactive in terminal 
if(interactive())fun()
##############################################################################
#################################### END #####################################
##############################################################################;

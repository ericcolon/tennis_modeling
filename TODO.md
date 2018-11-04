#### ELO Model
* Before 2003, no distinction in date between matches in same tournament! 
* Look into carpet vs other surfaces for indoor.
* Include fatigue
    * Include interaction(s) between age and fatigue
    * Age only in dataset without match dates :/
* Tune avg. step-size numerator...weightings affect step-size unless weights are normalized. 
    * Normalize all weights to avg to 1. and then tune c parameter.
* Weight by margin of victory
    * Something in between proportion of games won and just pure victory...like (prop games won by winner) * k, 0 < k < 1.
* NOTE: Results look better than they actually are, because I've removed retirements and walkovers from test sets
* Head-to-head features
* Figure out 3 vs. 5 set matches
    * Separate models with different weighting?
    * Multiplier times prediction that makes predictions closer to 50/50 for 3-setters?
    * Train probability of winning a set?

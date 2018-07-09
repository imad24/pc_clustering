def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

def MAPE(actual, estimate):
    '''Given two lists, one of actual values and one of estimated values, 
        computes the Mean Absolute Percentage Error'''
        
    if len(actual) != len(estimate):
        print("ERROR: Lists not the same length.")
        return []
        
    pcterrors = []
    
    for i in range(len(estimate)):
        pcterrors.append(abs(estimate[i]-actual[i])/actual[i])
    
    return sum(pcterrors)/len(pcterrors)

def holtwinters(ts, *args):
    '''Uses the Holt-Winters exp. smoothing method to forecast the next
       three points in a time series.  The second two arguments are 
       smoothing coefficients, alpha and beta.  If no coefficients are given,
       both are assumed to be 0.5.
       '''
       
    if len(args) >= 1:
        alpha = args[0]
       
    else:
        alpha = .5
        #findcoeff = True
    
    if len(args) >= 2:
        beta = args[1]
    else:
        beta = .5
            
    if len(ts) < 3:
        print("ERROR: At least three points are required for TS forecast.")
        return 0
    
    est = []    #estimated value (level)
    trend = []  #estimated trend
    
    '''For first value, assume trend and level are both 0.'''
    est.append(0)
    trend.append(0)
    
    '''For second value, assume trend still 0 and level same as first          
        actual value'''
    est.append(ts[0])
    trend.append(0)
    
    '''Now roll on for the rest of the values'''
    for i in range(len(ts)-2):
        trend.append(beta*(ts[i+1]-ts[i])+(1-beta)*trend[i+1])
        est.append(alpha*ts[i+1]+(1-alpha)*est[i+1]+trend[i+2])
        
    
    '''now back-cast for the first three values that we fudged'''
    est.reverse()
    trend.reverse()
    ts.reverse()
    
    for i in range(len(ts)-3, len(ts)):
        trend[i] = beta*(ts[i-1]-ts[i-2])+(1-beta)*(trend[i-1])
        est[i] = alpha*ts[i-1]+(1-alpha)*est[i-1]+trend[i]
    
       
    est.reverse()
    trend.reverse()
    ts.reverse()
    
    '''and do one last forward pass to smooth everything out'''
    for i in range(2, len(ts)):
        trend[i] = beta*(ts[i-1]-ts[i-2])+(1-beta)*(trend[i-1])
        est[i]= alpha*ts[i-1]+(1-alpha)*est[i-1]+trend[i]
        
    
    '''Holt-Winters method is only good for about 3 periods out'''
    next3 = [alpha*ts[-1]+(1-alpha)*(est[-1])+beta*(ts[-1]-ts[-2])+(1-beta)*         trend[-1]]
    next3.append(next3[0]+trend[-1])
    next3.append(next3[1]+trend[-1])
    
    return next3, MAPE(ts,est)


def holtwinters_auto(ts, *args):
    '''Calls the holtwinters function, but automatically determines the
    alpha and betta coefficients which minimize the error.
    
    The optional argument is the number of digits of precision you need
    for the coefficients.  The default is 4, which is plenty for most real
    life forecasting applications.
    '''
    
    if len(args) > 0:
        digits = args[0]
    else:
        digits = 4
    
    '''Perform an iterative grid search to find minimum MAPE'''
    
    alpha = .5
    beta = .5
    
    for d in range(1,digits):
        grid = []
        for b in [x * .1**d+beta for x in range(-5,6)]:
            for a in [x * .1**d+alpha for x in range(-5,6)]:
                grid.append(holtwinters(ts, a, b)[-1])
                if grid[-1]==min(grid):
                    alpha = a
                    beta = b
            
    next3, mape = holtwinters(ts, alpha, beta)
        
    return(next3, mape, alpha, beta)
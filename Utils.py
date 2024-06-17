
def cubic(y0, y1, y2, y3, mu):
    return (
        (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
        (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
        (-0.5*y0+0.5*y2)*mu +
        (y1))
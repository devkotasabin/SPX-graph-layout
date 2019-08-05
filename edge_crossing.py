import math

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(px, py, qx, qy, rx, ry):
 if (qx <= max(px, rx) and qx >= min(px, rx) and qy <= max(py, ry) and qy >= min(py, ry)):
  return True
 return False

def strictlyOnSegment(px, py, qx, qy, rx, ry):
 if (qx < max(px, rx) and qx > min(px, rx) and qy < max(py, ry) and qy > min(py, ry)):
  return True
 return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise

def orientation(px, py, qx, qy, rx, ry):
 # See http://www.geeksforgeeks.org/orientation-3-ordered-points/
 # for details of below formula.
 val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

 if (val == 0):return 0

 # clock or counterclock wise
 if (val > 0):
  return 1
 else:
  return 2

def yInt(x1, y1, x2, y2):
 if (y1 == y2):return y1
 return y1 - slope(x1, y1, x2, y2) * x1

def slope(x1, y1, x2, y2):
 #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2))
 if (x1 == x2):return False
 return (y1 - y2) / (x1 - x2)

# The main function that returns true if line segment 'p1q1'
# and 'p2q2' intersect.
def doSegmentsIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
 # Find the four orientations needed for general and
 # special cases
 o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y)
 o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y)
 o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y)
 o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y)

 #if(o1==0 or o2==0 or o3==0 or o4==0):return False
 # General case
 if (o1 != o2 and o3 != o4):
  return True

 # Special Cases
 # p1, q1 and p2 are colinear and p2 lies on segment p1q1
 if (o1 == 0 and onSegment(p1x, p1y, p2x, p2y, q1x, q1y)):return True

 # p1, q1 and p2 are colinear and q2 lies on segment p1q1
 if (o2 == 0 and onSegment(p1x, p1y, q2x, q2y, q1x, q1y)):return True

 # p2, q2 and p1 are colinear and p1 lies on segment p2q2
 if (o3 == 0 and onSegment(p2x, p2y, p1x, p1y, q2x, q2y)):return True

 # p2, q2 and q1 are colinear and q1 lies on segment p2q2
 if (o4 == 0 and onSegment(p2x, p2y, q1x, q1y, q2x, q2y)):return True

 return False # Doesn't fall in any of the above cases

def isSameCoord(x1, y1, x2, y2):
 if x1==x2 and y1==y2:
  return True
 return False

# do p is an end point of edge (u,v)
def isEndPoint(ux, uy, vx, vy, px, py):
 if isSameCoord(ux, uy, px, py) or isSameCoord(vx, vy, px, py):
  return True
 return False

# is (p1,q1) is adjacent to (p2,q2)?
def areEdgesAdjacent(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
 if isEndPoint(p1x, p1y, q1x, q1y, p2x, p2y):
  return True
 elif isEndPoint(p1x, p1y, q1x, q1y, q2x, q2y):
  return True
 return False

def isColinear(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
 x1 = p1x-q1x
 y1 = p1y-q1y
 x2 = p2x-q2x
 y2 = p2y-q2y
 cross_prod_value = x1*y2 - x2*y1
 if cross_prod_value==0:
  return True
 return False

# here p1q1 is one segment, and p2q2 is another
# this function checks first whether there is a shared vertex
# then it checks whether they are colinear
# finally it checks the segment intersection
def doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
 if areEdgesAdjacent(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
  if isColinear(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
   if strictlyOnSegment(p1x, p1y, p2x, p2y, q1x, q1y) or strictlyOnSegment(p1x, p1y, q2x, q2y, q1x, q1y) or strictlyOnSegment(p2x, p2y, p1x, p1y, q2x, q2y) or strictlyOnSegment(p2x, p2y, q1x, q1y, q2x, q2y):
    return True
   else:
    return False
  else:
   return False
 return doSegmentsIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)

def getIntersection(x11, y11, x12, y12, x21, y21, x22, y22):
 slope1 = 0
 slope2 = 0
 yint1 = 0
 yint2 = 0
 intx = 0
 inty = 0

 #TODO: Please Check all four cases
 if (x11 == x21 and y11 == y21):return [x11, y11]
 if (x12 == x22 and y12 == y22):return [x12, y22]
 # Check 1st point of edge 1 with 2nd point of edge 2 and viceversa

 slope1 = slope(x11, y11, x12, y12)
 slope2 = slope(x21, y21, x22, y22)
 #print('slope1:'+str(slope1))
 #print('slope2:'+str(slope2))
 if (slope1 == slope2):return False

 yint1 = yInt(x11, y11, x12, y12)
 yint2 = yInt(x21, y21, x22, y22)
 #print('yint1:'+str(yint1))
 #print('yint2:'+str(yint2))
 if (yint1 == yint2):
  if (yint1 == False):return False
  else:return [0, yint1]

 if(x11 == x12):return [x11, slope2*x11+yint2]
 if(x21 == x22):return [x21, slope1*x21+yint1]
 if(y11 == y12):return [(y11-yint2)/slope2,y11]
 if(y21 == y22):return [(y21-yint1)/slope1,y21]

 if (slope1 == False):return [y21, slope2 * y21 + yint2]
 if (slope2 == False):return [y11, slope1 * y11 + yint1]
 intx = (yint1 - yint2)/ (slope2-slope1)
 return [intx, slope1 * intx + yint1]

def to_deg(rad):
 return rad*180/math.pi

# x1,y1 is the 1st pt, x2,y2 is the 2nd pt, x3,y3 is the intersection pt
def getAngleLineSegDegree(x1,y1,x2,y2,x3,y3):
 #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2)+',x3:'+str(x3)+',y3:'+str(y3))
 # Uses dot product
 dc1x = x1-x3
 dc2x = x2-x3
 dc1y = y1-y3
 dc2y = y2-y3
 norm1 = math.sqrt(math.pow(dc1x,2) + math.pow(dc1y,2))
 norm2 = math.sqrt(math.pow(dc2x,2) + math.pow(dc2y,2))
 if norm1==0 or norm2==0:
  return -1
 angle = math.acos((dc1x*dc2x + dc1y*dc2y)/(norm1*norm2))
 # if angle > math.pi/2.0:
 #  angle = math.pi - angle
 #print('angle:'+str(angle))
 #return angle
 return to_deg(angle)

# x1,y1 is the 1st pt, x2,y2 is the 2nd pt, x3,y3 is the intersection pt
def getAngleLineSeg(x1,y1,x2,y2,x3,y3):
 #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2)+',x3:'+str(x3)+',y3:'+str(y3))
 # Uses dot product
 dc1x = x1-x3
 dc2x = x2-x3
 dc1y = y1-y3
 dc2y = y2-y3
 norm1 = math.sqrt(math.pow(dc1x,2) + math.pow(dc1y,2))
 norm2 = math.sqrt(math.pow(dc2x,2) + math.pow(dc2y,2))
 if norm1==0 or norm2==0:
  return -1
 angle = math.acos((dc1x*dc2x + dc1y*dc2y)/(norm1*norm2))
 # if angle > math.pi/2.0:
 #  angle = math.pi - angle
 #print('angle:'+str(angle))
 return angle

#test cases
#intersecting
#print(getIntersection(1, 1, 2, 2, 2, 1, 1, 2))
#not intersecting
#print(getIntersection(1, 1, 2, 2, 2, 1, 3, 2))
#intersecting
#print(getIntersection(2,1,2,2,2,1,3,2))

#test cases
#right angle
#print(getAngleLineSeg(1,2,2,1,1,1))
#half of right angle
#print(getAngleLineSeg(2,2,2,1,1,1))
#same direction
#print(getAngleLineSeg(2,1,3,1,1,1))
#opposite direction
#print(getAngleLineSeg(1,2,3,2,2,2))
#same angle
#print(getAngleLineSeg(2,1,3,1,1,1))

#test cases
#print(doIntersect(1,1,2,1,3,1,4,1))
#print(doIntersect(1,1,4,1,2,1,3,1))

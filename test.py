from heapq import nlargest 

test={"a":1, "b":2, "c":4, "d":9}

print(nlargest(2, test, key=test.get))
